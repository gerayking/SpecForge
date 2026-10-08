# coding=utf-8
"""DSpark training objective over selected target KV and compact teacher logits."""

from __future__ import annotations

from typing import Dict, Mapping, Tuple

import torch
import torch.distributed as dist
import torch.nn.functional as F

from specforge.algorithms.common.dflash_family_model import (
    FLEX_ATTENTION_AVAILABLE,
    OnlineDSparkModel,
    create_dflash_block_mask,
    create_dflash_sdpa_mask,
)
from specforge.core.chunking import checkpointed_chunk_reduce
from specforge.modeling.draft.target_kv import SharedHeadTransform, target_kv_contract


class OnlineTargetKVDSparkModel(OnlineDSparkModel):
    """Uses captured target KV as context and top-128 logits as teacher data."""

    input_mode = "target_kv"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        contract = target_kv_contract(self.draft_model.config)
        if contract is None:
            raise ValueError("target-KV DSpark model needs a target-KV draft")
        self.target_kv_contract = contract
        self.shared_head_transform = SharedHeadTransform.decode(
            contract["teacher"]["output_transform"]
        )
        self.lambda_tv = float(contract["training"]["lambda_tv"])
        if self.dspark_l1_loss_alpha != 0 or self.dspark_confidence_head_alpha != 0:
            raise ValueError(
                "target-KV DSpark uses CE+TV; hidden-state L1/confidence weights "
                "must be zero"
            )

    def _forward_target_kv_blocks(
        self,
        *,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        loss_mask: torch.Tensor,
        target_kv: Mapping[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        anchor_positions, block_keep_mask = self._sample_anchor_positions(
            seq_len, loss_mask, device
        )
        noise_embedding = self._create_noise_embed(
            input_ids, anchor_positions, block_keep_mask
        )
        target_hidden = self.draft_model.encode_target_kv(target_kv, position_ids)
        offsets = torch.arange(self.block_size, device=device).view(1, 1, -1)
        anchor_actual_positions = torch.gather(position_ids, 1, anchor_positions)
        draft_position_ids = (anchor_actual_positions.unsqueeze(-1) + offsets).reshape(
            batch_size, -1
        )
        full_position_ids = torch.cat([position_ids, draft_position_ids], dim=1)
        if self.attention_backend == "flex_attention":
            attention_mask = create_dflash_block_mask(
                anchor_positions=anchor_positions,
                block_keep_mask=block_keep_mask,
                S=seq_len,
                block_size=self.block_size,
                device=device,
            )
        else:
            attention_mask = create_dflash_sdpa_mask(
                anchor_positions=anchor_positions,
                block_keep_mask=block_keep_mask,
                S=seq_len,
                block_size=self.block_size,
                device=device,
            )
        output_hidden = self.draft_model(
            position_ids=full_position_ids,
            noise_embedding=noise_embedding,
            target_hidden=target_hidden,
            attention_mask=attention_mask,
        )
        return anchor_positions, block_keep_mask, output_hidden

    def _objective_chunk(
        self,
        hidden: torch.Tensor,
        prev_token_ids: torch.Tensor,
        target_ids: torch.Tensor,
        loss_weights: torch.Tensor,
        eval_mask: torch.Tensor,
        teacher_topk_ids: torch.Tensor,
        teacher_topk_logits: torch.Tensor,
        teacher_logsumexp: torch.Tensor,
    ) -> Tuple[torch.Tensor, ...]:
        batch_size, num_blocks, block_size, hidden_size = hidden.shape
        base_logits = self.shared_head_transform.apply(
            self.lm_head(
                hidden.reshape(batch_size, num_blocks * block_size, hidden_size)
            ).reshape(batch_size, num_blocks, block_size, -1)
        )
        draft_logits = self.draft_model.apply_logits_head(
            base_logits,
            prev_token_ids=prev_token_ids,
            hidden_states=hidden,
        ).float()
        vocab_size = draft_logits.shape[-1]
        if (teacher_topk_ids < 0).any() or (teacher_topk_ids >= vocab_size).any():
            raise ValueError(
                "teacher top-k IDs are outside the draft model vocabulary; "
                "target-KV v1 requires a shared global vocabulary"
            )
        ce = F.cross_entropy(
            draft_logits.reshape(-1, vocab_size),
            target_ids.reshape(-1),
            reduction="none",
        ).reshape_as(target_ids)
        teacher_probability = torch.exp(
            teacher_topk_logits.float() - teacher_logsumexp.float().unsqueeze(-1)
        )
        draft_logsumexp = torch.logsumexp(draft_logits, dim=-1, keepdim=True)
        draft_probability = torch.exp(
            torch.gather(draft_logits, -1, teacher_topk_ids.long()) - draft_logsumexp
        )
        tv = 0.5 * (teacher_probability - draft_probability).abs().sum(dim=-1)
        ce_num = (ce * loss_weights).sum()
        tv_num = (tv * loss_weights).sum()
        with torch.no_grad():
            predicted_ids = draft_logits.argmax(dim=-1)
            correct_num = ((predicted_ids == target_ids) & eval_mask).sum().float()
            eval_den = eval_mask.sum().float()
        return ce_num, tv_num, correct_num, eval_den

    def _compute_target_kv_loss(
        self,
        *,
        output_hidden: torch.Tensor,
        target_ids: torch.Tensor,
        eval_mask: torch.Tensor,
        prev_token_ids: torch.Tensor,
        safe_label_indices: torch.Tensor,
        teacher_topk_ids: torch.Tensor,
        teacher_topk_logits: torch.Tensor,
        teacher_logsumexp: torch.Tensor,
        teacher_valid: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, object]]:
        batch_size, num_blocks, block_size = target_ids.shape
        hidden = output_hidden.reshape(batch_size, num_blocks, block_size, -1)
        topk = teacher_topk_ids.shape[-1]
        gather_ids = safe_label_indices.unsqueeze(-1).expand(-1, -1, -1, topk)
        gathered_ids = torch.gather(
            teacher_topk_ids.unsqueeze(1).expand(-1, num_blocks, -1, -1),
            2,
            gather_ids,
        )
        gathered_logits = torch.gather(
            teacher_topk_logits.unsqueeze(1).expand(-1, num_blocks, -1, -1),
            2,
            gather_ids,
        )
        gathered_lse = torch.gather(
            teacher_logsumexp.unsqueeze(1).expand(-1, num_blocks, -1),
            2,
            safe_label_indices,
        )
        gathered_valid = torch.gather(
            teacher_valid.unsqueeze(1).expand(-1, num_blocks, -1),
            2,
            safe_label_indices,
        )
        eval_mask = eval_mask & gathered_valid.bool()
        loss_weights = self._dspark_loss_weight_mask(eval_mask)
        local_den = loss_weights.sum()
        ce_num, tv_num, correct_num, eval_den = checkpointed_chunk_reduce(
            self._objective_chunk,
            hidden,
            prev_token_ids,
            target_ids,
            loss_weights,
            eval_mask,
            gathered_ids,
            gathered_logits,
            gathered_lse,
            chunk_size=self.objective_chunk_blocks,
            dim=1,
        )
        global_den = local_den.detach().clone()
        world_size = 1
        if dist.is_available() and dist.is_initialized():
            world_size = dist.get_world_size()
            if world_size > 1:
                dist.all_reduce(global_den, op=dist.ReduceOp.SUM)
        if float(global_den) <= 0:
            raise ValueError("target-KV DSpark batch has no teacher-supervised tokens")
        loss = world_size * (ce_num + self.lambda_tv * tv_num) / global_den
        accuracy = correct_num / eval_den.clamp_min(1.0)
        return loss, {
            "accuracy": accuracy.detach(),
            "accuracy_denom": eval_den.detach(),
            "ce_loss": (ce_num / local_den.clamp_min(1.0)).detach(),
            "tv_loss": (tv_num / local_den.clamp_min(1.0)).detach(),
            "ratio_metrics": {
                "accuracy": (correct_num.detach(), eval_den.detach()),
                "ce_loss": (ce_num.detach(), local_den.detach()),
                "tv_loss": (tv_num.detach(), local_den.detach()),
            },
        }

    def forward(
        self,
        *,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        loss_mask: torch.Tensor,
        target_kv: Mapping[str, torch.Tensor],
        teacher_topk_ids: torch.Tensor,
        teacher_topk_logits: torch.Tensor,
        teacher_logsumexp: torch.Tensor,
        teacher_valid: torch.Tensor,
    ):
        if self.attention_backend == "flex_attention" and not FLEX_ATTENTION_AVAILABLE:
            raise ValueError(
                "flex_attention is unavailable on this device; use sdpa/eager"
            )
        anchor_positions, block_keep_mask, output_hidden = (
            self._forward_target_kv_blocks(
                input_ids=input_ids,
                position_ids=position_ids,
                loss_mask=loss_mask,
                target_kv=target_kv,
            )
        )
        target_ids, eval_mask, safe_label_indices = self._build_dspark_labels_and_mask(
            input_ids, loss_mask, anchor_positions, block_keep_mask
        )
        anchor_token_ids = torch.gather(input_ids, 1, anchor_positions)
        prev_token_ids = torch.cat(
            [anchor_token_ids.unsqueeze(-1), target_ids[:, :, :-1]], dim=-1
        )
        loss, metrics = self._compute_target_kv_loss(
            output_hidden=output_hidden,
            target_ids=target_ids,
            eval_mask=eval_mask,
            prev_token_ids=prev_token_ids,
            safe_label_indices=safe_label_indices,
            teacher_topk_ids=teacher_topk_ids,
            teacher_topk_logits=teacher_topk_logits,
            teacher_logsumexp=teacher_logsumexp,
            teacher_valid=teacher_valid,
        )
        return loss, metrics.pop("accuracy"), metrics


__all__ = ["OnlineTargetKVDSparkModel"]
