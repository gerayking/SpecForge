# coding=utf-8
"""Align target-KV samples and either teacher representation to token positions."""

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
    hidden = "teacher_final_hidden" in raw
    teacher_names = (
        {"teacher_final_hidden"}
        if hidden
        else {"teacher_topk_ids", "teacher_topk_logits", "teacher_logsumexp"}
    )
    if hidden and any(
        name in raw
        for name in ("teacher_topk_ids", "teacher_topk_logits", "teacher_logsumexp")
    ):
        raise ValueError("sample mixes hidden and top-k teacher representations")
    required_aux = {
        "token_ids",
        "position_ids",
        "loss_mask",
        "kv_valid",
        "logits_positions",
    } | teacher_names
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

    teacher_valid = torch.zeros(total_length, dtype=torch.bool)
    teacher_valid[logits_positions] = True
    teacher = {}
    for name in teacher_names:
        value = raw[name]
        dtype = torch.long if name == "teacher_topk_ids" else value.dtype
        aligned = torch.zeros((total_length, *value.shape[1:]), dtype=dtype)
        aligned[logits_positions] = value[keep].to(dtype)
        teacher[name] = aligned.unsqueeze(0)

    normalized = {
        "input_ids": raw["token_ids"][:total_length].long().unsqueeze(0),
        "position_ids": raw["position_ids"][:total_length].long().unsqueeze(0),
        "loss_mask": raw["loss_mask"][:total_length].long().unsqueeze(0),
        "kv_valid": raw["kv_valid"][:total_length].bool().unsqueeze(0),
        **teacher,
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
        "teacher_valid",
        *kv_names,
    )

    def collate(features):
        hidden = ["teacher_final_hidden" in item for item in features]
        if any(hidden) and not all(hidden):
            raise ValueError("batch mixes hidden and top-k teacher representations")
        teacher_names = (
            ("teacher_final_hidden",)
            if all(hidden)
            else ("teacher_topk_ids", "teacher_topk_logits", "teacher_logsumexp")
        )
        keys = (*required, *teacher_names)
        return pad_and_concatenate_features(
            features,
            sequence_axes={name: 1 for name in keys},
            required_keys=keys,
        )

    return collate


__all__ = [
    "build_target_kv_collator",
    "build_target_kv_normalizer",
    "normalize_target_kv_sample",
    "target_kv_feature_names",
]
