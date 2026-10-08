# coding=utf-8
"""Normalization and collation for ``maas_target_kv_v1`` samples."""

from __future__ import annotations

from functools import partial
from typing import Any, Mapping

import torch

from specforge.algorithms.common.collation import pad_and_concatenate_features


def target_kv_feature_names(contract: Mapping[str, Any]) -> tuple[str, ...]:
    return tuple(
        f"target_{component}.{layer['layer_id']}"
        for layer in contract["kv"]["layers"]
        for component in ("k", "v")
    )


def normalize_target_kv_sample(
    raw: Mapping[str, torch.Tensor],
    *,
    contract: Mapping[str, Any],
    max_len: int,
) -> dict[str, torch.Tensor]:
    required_aux = {
        "token_ids",
        "position_ids",
        "loss_mask",
        "kv_valid",
        "logits_positions",
        "teacher_topk_ids",
        "teacher_topk_logits",
        "teacher_logsumexp",
    }
    kv_names = set(target_kv_feature_names(contract))
    missing = (required_aux | kv_names) - set(raw)
    if missing:
        raise KeyError(f"target-KV sample is missing tensors {sorted(missing)}")
    total_length = min(int(raw["token_ids"].numel()), int(max_len))
    if total_length < 2:
        raise ValueError("target-KV sample needs at least two tokens")
    logits_positions = raw["logits_positions"].long()
    keep = logits_positions < total_length
    logits_positions = logits_positions[keep]
    if not keep.any():
        raise ValueError("target-KV sample has no teacher rows after truncation")

    topk = int(raw["teacher_topk_ids"].shape[-1])
    teacher_ids = torch.zeros(total_length, topk, dtype=torch.long)
    teacher_logits = torch.zeros(total_length, topk, dtype=torch.float32)
    teacher_lse = torch.zeros(total_length, dtype=torch.float32)
    teacher_valid = torch.zeros(total_length, dtype=torch.bool)
    teacher_ids[logits_positions] = raw["teacher_topk_ids"][keep].long()
    teacher_logits[logits_positions] = raw["teacher_topk_logits"][keep].float()
    teacher_lse[logits_positions] = raw["teacher_logsumexp"][keep].float()
    teacher_valid[logits_positions] = True

    normalized = {
        "input_ids": raw["token_ids"][:total_length].long().unsqueeze(0),
        "position_ids": raw["position_ids"][:total_length].long().unsqueeze(0),
        "loss_mask": raw["loss_mask"][:total_length].long().unsqueeze(0),
        "kv_valid": raw["kv_valid"][:total_length].bool().unsqueeze(0),
        "teacher_topk_ids": teacher_ids.unsqueeze(0),
        "teacher_topk_logits": teacher_logits.unsqueeze(0),
        "teacher_logsumexp": teacher_lse.unsqueeze(0),
        "teacher_valid": teacher_valid.unsqueeze(0),
    }
    for name in kv_names:
        value = raw[name]
        if value.ndim != 3:
            raise ValueError(
                f"target-KV tensor {name!r} must have [tokens, heads, dim] shape"
            )
        normalized[name] = value[:total_length].unsqueeze(0)
    return normalized


def build_target_kv_normalizer(contract: Mapping[str, Any], max_len: int):
    return partial(
        normalize_target_kv_sample,
        contract=contract,
        max_len=max_len,
    )


def build_target_kv_collator(contract: Mapping[str, Any]):
    kv_names = target_kv_feature_names(contract)
    required = (
        "input_ids",
        "position_ids",
        "loss_mask",
        "kv_valid",
        "teacher_topk_ids",
        "teacher_topk_logits",
        "teacher_logsumexp",
        "teacher_valid",
        *kv_names,
    )
    sequence_axes = {name: 1 for name in required}

    def collate(features):
        return pad_and_concatenate_features(
            features,
            sequence_axes=sequence_axes,
            required_keys=required,
        )

    return collate


__all__ = [
    "build_target_kv_collator",
    "build_target_kv_normalizer",
    "normalize_target_kv_sample",
    "target_kv_feature_names",
]
