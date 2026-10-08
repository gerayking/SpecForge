# coding=utf-8
"""Target-KV feature contract and trainable encoder shared with SGLang serving."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from typing import Any, Optional

import torch
import torch.nn.functional as F
from torch import nn


class TargetKVContractError(ValueError):
    pass


class SharedHeadTransform:
    """Frozen target-head scaling applied before the trainable Markov head."""

    def __init__(self, logit_scale=None, final_logit_softcapping=None):
        self.logit_scale = logit_scale
        self.final_logit_softcapping = final_logit_softcapping

    @classmethod
    def decode(cls, value: Any) -> "SharedHeadTransform":
        if value == "identity":
            return cls()
        if not isinstance(value, str):
            raise TargetKVContractError("teacher output_transform must be a string")
        try:
            raw = json.loads(value)
        except json.JSONDecodeError as exc:
            raise TargetKVContractError("invalid teacher output_transform") from exc
        if not isinstance(raw, dict) or set(raw) - {
            "logit_scale",
            "final_logit_softcapping",
        }:
            raise TargetKVContractError("unsupported teacher output_transform")
        scale = raw.get("logit_scale")
        softcap = raw.get("final_logit_softcapping")
        for name, item in (("logit_scale", scale), ("softcap", softcap)):
            if item is not None and (
                isinstance(item, bool)
                or not isinstance(item, (int, float))
                or not math.isfinite(item)
            ):
                raise TargetKVContractError(f"invalid teacher {name}")
        if softcap is not None and softcap < 0:
            raise TargetKVContractError("teacher softcap must be non-negative")
        return cls(scale, softcap)

    def apply(self, logits: torch.Tensor) -> torch.Tensor:
        logits = logits.float()
        if self.logit_scale is not None:
            logits = logits * self.logit_scale
        if self.final_logit_softcapping:
            cap = self.final_logit_softcapping
            logits = cap * torch.tanh(logits / cap)
        return logits


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def target_kv_contract(config: Any) -> Optional[dict]:
    raw = config if isinstance(config, dict) else config.to_dict()
    input_mode = raw.get("input_mode", "target_hidden")
    if input_mode == "target_hidden":
        if "target_kv_contract" in raw:
            raise TargetKVContractError(
                "hidden-input checkpoint carries a target-KV contract"
            )
        return None
    if input_mode != "target_kv":
        raise TargetKVContractError(f"unknown DSpark input_mode {input_mode!r}")
    if raw.get("architectures") != ["DSparkTargetKVDraftModel"]:
        raise TargetKVContractError("target-KV input requires DSparkTargetKVDraftModel")
    contract = raw.get("target_kv_contract")
    if not isinstance(contract, dict):
        raise TargetKVContractError("target-KV checkpoint has no contract")
    validate_target_kv_contract(contract)
    if raw.get("hidden_size") != contract["encoder"]["hidden_size"]:
        raise TargetKVContractError("KV encoder output and draft hidden size disagree")
    if raw.get("vocab_size") != contract["teacher"]["vocab_size"]:
        raise TargetKVContractError(
            "draft vocabulary and bound teacher vocabulary disagree"
        )
    if (
        raw.get("block_size") != contract["sequence"]["prediction_count"]
        or raw.get("mask_token_id") != contract["sequence"]["mask_token_id"]
    ):
        raise TargetKVContractError(
            "serving block or mask token differs from the target-KV contract"
        )
    if raw.get("enable_confidence_head") is not False:
        raise TargetKVContractError(
            "target-KV v1 requires enable_confidence_head=false"
        )
    markov_rank = raw.get("markov_rank")
    if (
        isinstance(markov_rank, bool)
        or not isinstance(markov_rank, int)
        or markov_rank < 1
        or raw.get("markov_head_type", "vanilla") not in ("vanilla", "gated", "rnn")
    ):
        raise TargetKVContractError("invalid serving-compatible Markov head")
    return contract


def validate_target_kv_contract(contract: Mapping[str, Any]) -> None:
    required = {
        "teacher",
        "kv",
        "encoder",
        "sequence",
        "training",
        "compatibility",
        "validation",
        "architecture_revision",
        "input_mode",
        "schema_version",
        "feature_order",
        "feature_k_stage",
    }
    if set(contract) != required:
        raise TargetKVContractError("target-KV contract fields do not match v1")
    if (
        contract["architecture_revision"] != "dspark_target_kv_v1"
        or contract["input_mode"] != "target_kv"
        or contract["schema_version"] != 1
        or contract["feature_order"] != "layer_then_k_v_head_dim"
    ):
        raise TargetKVContractError("unsupported target-KV contract version")
    teacher = contract["teacher"]
    kv = contract["kv"]
    encoder = contract["encoder"]
    sequence = contract["sequence"]
    training = contract["training"]
    if not all(
        isinstance(value, dict) for value in (teacher, kv, encoder, sequence, training)
    ):
        raise TargetKVContractError("target-KV contract sections must be objects")
    if teacher.get("adapter_revision") is not None:
        raise TargetKVContractError("target-KV v1 does not support target adapters")
    SharedHeadTransform.decode(teacher.get("output_transform"))
    if not isinstance(teacher.get("vocab_size"), int) or teacher["vocab_size"] < 128:
        raise TargetKVContractError("invalid teacher vocabulary")
    layers = kv.get("layers")
    selected = kv.get("selected_layer_ids")
    if (
        not isinstance(layers, list)
        or not layers
        or not isinstance(selected, list)
        or [layer.get("layer_id") for layer in layers] != selected
        or len(set(selected)) != len(selected)
    ):
        raise TargetKVContractError("invalid target-KV layer geometry")
    if kv.get("dtype") not in ("bfloat16", "float16"):
        raise TargetKVContractError("target-KV dtype must be BF16 or FP16")
    if kv.get("source_k_stage") not in ("pre_rope", "post_rope"):
        raise TargetKVContractError("unknown target K stage")
    if contract["feature_k_stage"] not in ("pre_rope", "post_rope"):
        raise TargetKVContractError("unknown KV encoder feature K stage")
    if (
        contract["feature_k_stage"] == "post_rope"
        and kv["source_k_stage"] != "post_rope"
    ):
        raise TargetKVContractError("post-RoPE features need post-RoPE source K")
    rope = kv.get("rope_config")
    if not isinstance(rope, dict):
        raise TargetKVContractError("KV contract has no resolved RoPE config")
    if hashlib.sha256(_canonical_bytes(rope)).hexdigest() != kv.get(
        "rope_config_sha256"
    ):
        raise TargetKVContractError("KV RoPE digest mismatch")
    if (
        rope.get("type") != "default"
        or rope.get("scaling") is not None
        or not isinstance(rope.get("rotary_dim"), int)
        or rope["rotary_dim"] < 2
        or rope["rotary_dim"] % 2
        or not isinstance(rope.get("theta"), (int, float))
        or not math.isfinite(rope["theta"])
        or rope["theta"] <= 0
        or not isinstance(rope.get("interleaved"), bool)
    ):
        raise TargetKVContractError("unsupported target-KV RoPE contract")
    for layer in layers:
        for name in ("num_kv_heads", "key_head_dim", "value_head_dim"):
            if not isinstance(layer.get(name), int) or layer[name] < 1:
                raise TargetKVContractError("invalid target-KV layer dimension")
        if layer["key_head_dim"] < rope["rotary_dim"]:
            raise TargetKVContractError("rotary dimension exceeds K head dimension")
    if (
        not isinstance(encoder.get("hidden_size"), int)
        or encoder["hidden_size"] < 1
        or encoder.get("bias") is not False
        or encoder.get("norm_semantics") != "fp32_variance_and_weight_then_cast"
        or not isinstance(encoder.get("rms_norm_eps"), (int, float))
        or not math.isfinite(encoder["rms_norm_eps"])
        or encoder["rms_norm_eps"] <= 0
    ):
        raise TargetKVContractError("invalid target-KV encoder contract")
    prediction_count = sequence.get("prediction_count")
    if (
        not isinstance(prediction_count, int)
        or not 1 <= prediction_count <= 64
        or sequence.get("input_length") != prediction_count
        or sequence.get("context_boundary") != "strictly_before_anchor"
        or sequence.get("backbone_input") != "anchor_then_masks"
        or sequence.get("label_shift") != 1
        or sequence.get("markov_previous") != "anchor_then_previous_labels"
        or sequence.get("position_semantics") != "actual_target_positions"
        or not isinstance(sequence.get("mask_token_id"), int)
        or not 0 <= sequence["mask_token_id"] < teacher["vocab_size"]
    ):
        raise TargetKVContractError("invalid target-KV sequence contract")
    if (
        training.get("objective") != "ce_tv128_full_vocabulary_lse_t1_v1"
        or training.get("tv_tail_policy") != "omit_tail_without_renormalization"
        or training.get("confidence_policy") != "disabled"
        or training.get("shared_modules") != "frozen_target_reference"
        or training.get("shared_head_transform_placement") != "before_markov"
        or training.get("base_logits_dtype") != "float32"
        or not isinstance(training.get("lambda_tv"), (int, float))
        or not math.isfinite(training["lambda_tv"])
        or training["lambda_tv"] < 0
    ):
        raise TargetKVContractError("unsupported target-KV training objective")


def target_kv_feature_size(contract: Mapping[str, Any]) -> int:
    return sum(
        layer["num_kv_heads"] * (layer["key_head_dim"] + layer["value_head_dim"])
        for layer in contract["kv"]["layers"]
    )


def _inverse_standard_rope(
    key: torch.Tensor, positions: torch.Tensor, rope: Mapping[str, Any]
) -> torch.Tensor:
    if key.ndim != 4 or positions.shape != key.shape[:2]:
        raise TargetKVContractError(
            "batched K and position IDs must cover the same token rows"
        )
    rotary_dim = int(rope["rotary_dim"])
    frequencies = 1.0 / (
        float(rope["theta"])
        ** (
            torch.arange(
                0,
                rotary_dim,
                2,
                dtype=torch.float32,
                device=key.device,
            )
            / rotary_dim
        )
    )
    angles = positions.float().unsqueeze(-1) * frequencies
    cos = angles.cos().unsqueeze(-2)
    sin = angles.sin().unsqueeze(-2)
    source = key.float()
    rotated = source[..., :rotary_dim]
    if rope["interleaved"]:
        left, right = rotated[..., 0::2], rotated[..., 1::2]
        restored = torch.stack(
            (left * cos + right * sin, right * cos - left * sin), dim=-1
        ).flatten(-2)
    else:
        left, right = rotated.chunk(2, dim=-1)
        restored = torch.cat(
            (left * cos + right * sin, right * cos - left * sin), dim=-1
        )
    return torch.cat((restored, source[..., rotary_dim:]), dim=-1)


def target_kv_features(
    contract: Mapping[str, Any],
    tensors: Mapping[str, torch.Tensor],
    positions: torch.Tensor,
) -> torch.Tensor:
    kv = contract["kv"]
    expected_names = {
        f"target_{component}.{layer['layer_id']}"
        for layer in kv["layers"]
        for component in ("k", "v")
    }
    if set(tensors) != expected_names:
        raise TargetKVContractError(
            "target-KV tensors do not match the selected layer contract"
        )
    expected_dtype = torch.bfloat16 if kv["dtype"] == "bfloat16" else torch.float16
    features = []
    for layer in kv["layers"]:
        for component, dimension in (
            ("k", layer["key_head_dim"]),
            ("v", layer["value_head_dim"]),
        ):
            value = tensors[f"target_{component}.{layer['layer_id']}"]
            expected_shape = (
                positions.shape[0],
                positions.shape[1],
                layer["num_kv_heads"],
                dimension,
            )
            if (
                tuple(value.shape) != expected_shape
                or value.dtype != expected_dtype
                or value.device != positions.device
            ):
                raise TargetKVContractError(
                    "target-KV tensor disagrees with its feature contract"
                )
            value = value.detach()
            if (
                component == "k"
                and kv["source_k_stage"] == "post_rope"
                and contract["feature_k_stage"] == "pre_rope"
            ):
                value = _inverse_standard_rope(value, positions, kv["rope_config"])
            features.append(value.float().flatten(2))
    return torch.cat(features, dim=-1)


class TargetKVContextEncoder(nn.Module):
    def __init__(self, contract: Mapping[str, Any]):
        super().__init__()
        validate_target_kv_contract(contract)
        self.contract = dict(contract)
        encoder = contract["encoder"]
        self.projection = nn.Linear(
            target_kv_feature_size(contract), encoder["hidden_size"], bias=False
        )
        self.norm_weight = nn.Parameter(torch.ones(encoder["hidden_size"]))

    def forward(
        self, tensors: Mapping[str, torch.Tensor], positions: torch.Tensor
    ) -> torch.Tensor:
        features = target_kv_features(self.contract, tensors, positions)
        hidden = F.linear(
            features.to(self.projection.weight.dtype), self.projection.weight
        )
        value = hidden.float()
        variance = value.square().mean(dim=-1, keepdim=True)
        value = value * torch.rsqrt(
            variance + float(self.contract["encoder"]["rms_norm_eps"])
        )
        return (value * self.norm_weight.float()).to(hidden.dtype)


__all__ = [
    "SharedHeadTransform",
    "TargetKVContextEncoder",
    "TargetKVContractError",
    "target_kv_contract",
    "target_kv_feature_size",
    "target_kv_features",
    "validate_target_kv_contract",
]
