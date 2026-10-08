# coding=utf-8
"""Trainable DSpark architecture that consumes selected target KV tensors."""

from __future__ import annotations

from typing import Optional

import torch
from transformers.cache_utils import Cache
from transformers.modeling_outputs import CausalLMOutputWithPast

from .dspark import DSparkDraftModel
from .registry import register_draft
from .target_kv import TargetKVContextEncoder, target_kv_contract


@register_draft
class DSparkTargetKVDraftModel(DSparkDraftModel):
    """Checkpoint-compatible counterpart of SGLang's target-KV draft model."""

    def __init__(self, config) -> None:
        contract = target_kv_contract(config)
        if contract is None:
            raise ValueError("DSparkTargetKVDraftModel requires input_mode=target_kv")
        method_config = dict(getattr(config, "dflash_config", None) or {})
        serving_fields = {
            "markov_rank": getattr(config, "markov_rank", None),
            "markov_head_type": getattr(config, "markov_head_type", "vanilla"),
            "enable_confidence_head": getattr(config, "enable_confidence_head", None),
            "confidence_head_with_markov": getattr(
                config, "confidence_head_with_markov", False
            ),
        }
        for name, value in serving_fields.items():
            if name in method_config and method_config[name] != value:
                raise ValueError(f"top-level and dflash_config {name} fields disagree")
            method_config[name] = value
        method_config.update(
            {
                "projector_type": "dspark",
                "target_layer_ids": list(contract["kv"]["selected_layer_ids"]),
                "mask_token_id": int(contract["sequence"]["mask_token_id"]),
                "confidence_head_alpha": 0.0,
                "shift_label": True,
            }
        )
        config.dflash_config = method_config
        super().__init__(config)
        self.target_kv_contract = contract
        del self.fc
        del self.hidden_norm
        self.kv_encoder = TargetKVContextEncoder(contract)
        self.target_layer_ids = list(contract["kv"]["selected_layer_ids"])
        self.mask_token_id = int(contract["sequence"]["mask_token_id"])
        if self.enable_confidence_head or self.confidence_head is not None:
            raise ValueError("target-KV DSpark must disable its confidence head")
        if self.block_size != int(contract["sequence"]["prediction_count"]):
            raise ValueError("draft block_size and target-KV prediction_count disagree")

    def encode_target_kv(self, tensors, positions):
        return self.kv_encoder(tensors, positions)

    def forward(
        self,
        position_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        noise_embedding: Optional[torch.Tensor] = None,
        target_hidden: Optional[torch.Tensor] = None,
        past_key_values: Optional[Cache] = None,
        use_cache: bool = False,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        if target_hidden is None or target_hidden.shape[-1] != self.config.hidden_size:
            raise ValueError("target-KV draft requires encoded context hidden states")
        hidden_states = noise_embedding
        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        for layer in self.layers:
            hidden_states = layer(
                hidden_states=hidden_states,
                target_hidden=target_hidden,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
                use_cache=use_cache,
                position_embeddings=position_embeddings,
                **kwargs,
            )
        return self.norm(hidden_states)


__all__ = ["DSparkTargetKVDraftModel"]
