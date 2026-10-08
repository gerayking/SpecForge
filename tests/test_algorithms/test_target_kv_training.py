# coding=utf-8
"""Loss tests for compact teacher supervision in target-KV DSpark."""

import unittest
import copy

import torch
import torch.nn.functional as F
from torch import nn
from transformers import Qwen3Config

from specforge.algorithms.common.target_kv_model import OnlineTargetKVDSparkModel
from specforge.modeling.draft.dspark_target_kv import DSparkTargetKVDraftModel
from specforge.modeling.draft.target_kv import SharedHeadTransform
from tests.test_runtime.test_training_snapshot import _snapshot


class _IdentityDraft(nn.Module):
    def apply_logits_head(self, logits, *, prev_token_ids, hidden_states):
        del prev_token_ids, hidden_states
        return logits


def _objective_model():
    model = OnlineTargetKVDSparkModel.__new__(OnlineTargetKVDSparkModel)
    nn.Module.__init__(model)
    model.lm_head = nn.Linear(4, 256, bias=False)
    model.draft_model = _IdentityDraft()
    model.shared_head_transform = SharedHeadTransform.decode("identity")
    return model


def _target_kv_contract():
    _payloads, _publication, _tensors, partial = _snapshot()
    contract = copy.deepcopy(partial)
    contract.update(
        {
            "encoder": {
                "hidden_size": 8,
                "rms_norm_eps": 1e-6,
                "bias": False,
                "norm_semantics": "fp32_variance_and_weight_then_cast",
            },
            "sequence": {
                "prediction_count": 3,
                "mask_token_id": 255,
                "input_length": 3,
                "context_boundary": "strictly_before_anchor",
                "backbone_input": "anchor_then_masks",
                "label_shift": 1,
                "markov_previous": "anchor_then_previous_labels",
                "position_semantics": "actual_target_positions",
            },
            "training": {
                "lambda_tv": 0.5,
                "objective": "ce_tv128_full_vocabulary_lse_t1_v1",
                "tv_tail_policy": "omit_tail_without_renormalization",
                "confidence_policy": "disabled",
                "shared_modules": "frozen_target_reference",
                "shared_head_transform_placement": "before_markov",
                "base_logits_dtype": "float32",
            },
            "compatibility": {
                "sglang_revision": "test",
                "specforge_revision": "test",
                "serving_contract": "sglang_dspark_target_kv_v1",
            },
            "validation": {
                "golden_fixture_sha256": "3" * 64,
                "parity_rtol": 1e-5,
                "parity_atol": 1e-5,
            },
            "architecture_revision": "dspark_target_kv_v1",
            "schema_version": 1,
            "feature_order": "layer_then_k_v_head_dim",
            "feature_k_stage": "pre_rope",
        }
    )
    return contract


class TestTargetKVObjective(unittest.TestCase):
    def test_shared_head_transform_matches_serving_order(self):
        transform = SharedHeadTransform.decode(
            '{"logit_scale":0.5,"final_logit_softcapping":2.0}'
        )
        logits = torch.tensor([-8.0, -1.0, 0.0, 3.0, 8.0])
        expected = 2.0 * torch.tanh((logits.float() * 0.5) / 2.0)
        torch.testing.assert_close(transform.apply(logits), expected)

    def test_top128_tv_uses_full_vocabulary_normalization(self):
        torch.manual_seed(7)
        model = _objective_model()
        hidden = torch.randn(1, 1, 2, 4)
        logits = model.lm_head(hidden.reshape(1, 2, 4)).reshape(1, 1, 2, 256)
        teacher_logits, teacher_ids = logits.float().topk(128, dim=-1)
        teacher_lse = torch.logsumexp(logits.float(), dim=-1)
        target_ids = torch.tensor([[[3, 5]]])
        weights = torch.ones(1, 1, 2)
        ce_num, tv_num, correct_num, eval_den = model._objective_chunk(
            hidden,
            torch.tensor([[[1, 3]]]),
            target_ids,
            weights,
            weights.bool(),
            teacher_ids,
            teacher_logits,
            teacher_lse,
        )
        expected_ce = F.cross_entropy(
            logits.reshape(-1, 256), target_ids.reshape(-1), reduction="sum"
        )
        torch.testing.assert_close(ce_num, expected_ce)
        torch.testing.assert_close(tv_num, torch.zeros_like(tv_num), atol=1e-6, rtol=0)
        self.assertEqual(eval_den.item(), 2)
        self.assertGreaterEqual(correct_num.item(), 0)

    def test_teacher_ids_must_fit_draft_vocabulary(self):
        model = _objective_model()
        with self.assertRaisesRegex(ValueError, "shared global vocabulary"):
            model._objective_chunk(
                torch.zeros(1, 1, 1, 4),
                torch.zeros(1, 1, 1, dtype=torch.long),
                torch.zeros(1, 1, 1, dtype=torch.long),
                torch.ones(1, 1, 1),
                torch.ones(1, 1, 1, dtype=torch.bool),
                torch.full((1, 1, 1, 128), 256, dtype=torch.long),
                torch.zeros(1, 1, 1, 128),
                torch.ones(1, 1, 1),
            )

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_cuda_forward_backward_reaches_all_trainable_components(self):
        contract = _target_kv_contract()
        config = Qwen3Config(
            hidden_size=8,
            intermediate_size=16,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=4,
            vocab_size=256,
            max_position_embeddings=128,
            num_target_layers=2,
            block_size=3,
            mask_token_id=255,
            markov_rank=4,
            markov_head_type="vanilla",
            enable_confidence_head=False,
            confidence_head_with_markov=False,
            architectures=["DSparkTargetKVDraftModel"],
            input_mode="target_kv",
            target_kv_contract=contract,
        )
        config._attn_implementation = "sdpa"
        draft = DSparkTargetKVDraftModel(config).cuda().to(torch.bfloat16)
        embedding = nn.Embedding(256, 8).cuda().to(torch.bfloat16)
        head = nn.Linear(8, 256, bias=False).cuda().to(torch.bfloat16)
        embedding.requires_grad_(False)
        head.requires_grad_(False)
        model = OnlineTargetKVDSparkModel(
            draft_model=draft,
            target_lm_head=head,
            target_embed_tokens=embedding,
            mask_token_id=255,
            block_size=3,
            attention_backend="sdpa",
            num_anchors=2,
            loss_decay_gamma=None,
            dspark_ce_loss_alpha=1.0,
            dspark_l1_loss_alpha=0.0,
            dspark_confidence_head_alpha=0.0,
            objective_chunk_blocks=1,
        )
        length = 6
        topk_ids = torch.arange(128, device="cuda").view(1, 1, 128)
        topk_ids = topk_ids.expand(1, length, -1)
        topk_logits = torch.linspace(0, -5, 128, device="cuda").view(1, 1, 128)
        topk_logits = topk_logits.expand(1, length, -1)
        loss, _accuracy, _metrics = model(
            input_ids=torch.tensor([[1, 2, 3, 4, 5, 6]], device="cuda"),
            position_ids=torch.arange(length, device="cuda").view(1, -1),
            loss_mask=torch.tensor(
                [[0, 0, 1, 1, 1, 1]], device="cuda", dtype=torch.float32
            ),
            target_kv={
                "target_k.1": torch.randn(
                    1, length, 2, 4, device="cuda", dtype=torch.bfloat16
                ),
                "target_v.1": torch.randn(
                    1, length, 2, 4, device="cuda", dtype=torch.bfloat16
                ),
            },
            teacher_topk_ids=topk_ids,
            teacher_topk_logits=topk_logits,
            teacher_logsumexp=torch.full((1, length), 7.0, device="cuda"),
            teacher_valid=torch.tensor(
                [[0, 0, 1, 1, 1, 1]], device="cuda", dtype=torch.bool
            ),
        )
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        groups = {
            "kv_encoder": draft.kv_encoder.parameters(),
            "layers": draft.layers.parameters(),
            "markov_head": draft.markov_head.parameters(),
        }
        for name, parameters in groups.items():
            gradients = [parameter.grad for parameter in parameters]
            self.assertTrue(
                any(
                    gradient is not None
                    and torch.isfinite(gradient).all()
                    and gradient.abs().sum() > 0
                    for gradient in gradients
                ),
                name,
            )


if __name__ == "__main__":
    unittest.main()
