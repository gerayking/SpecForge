# coding=utf-8
"""Contract and transport tests for SGLang target-KV snapshots."""

import ctypes
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import torch

from specforge.algorithms.common.target_kv_data import (
    build_target_kv_collator,
    normalize_target_kv_sample,
)
from specforge.runtime.data_plane.training_snapshot import (
    MooncakeSnapshotFeatureStore,
    SnapshotContractError,
    SnapshotRefReader,
)


def _tensor_bytes(tensor):
    return bytes(memoryview(tensor.contiguous().view(torch.uint8).numpy()).cast("B"))


def _sha256(data):
    return hashlib.sha256(data).hexdigest()


def _snapshot():
    prompt_length, response_length = 2, 3
    total_length = prompt_length + response_length
    prefix = "draft-data/test-dataset/test-sample/g1/"
    rope = {
        "type": "default",
        "theta": 10000.0,
        "rotary_dim": 4,
        "interleaved": False,
        "scaling": None,
    }
    logits = torch.arange(128, 0, -1, dtype=torch.float32).repeat(response_length, 1)
    tensors = {
        "token_ids": torch.arange(total_length, dtype=torch.int32) + 3,
        "position_ids": torch.arange(total_length, dtype=torch.int64),
        "loss_mask": torch.tensor([0, 0, 1, 1, 1], dtype=torch.uint8),
        "kv_valid": torch.tensor([1, 1, 1, 1, 0], dtype=torch.uint8),
        "logits_positions": torch.arange(2, 5, dtype=torch.int32),
        "teacher_topk_ids": torch.arange(128, dtype=torch.int32).repeat(
            response_length, 1
        ),
        "teacher_topk_logits": logits,
        "teacher_logsumexp": torch.logsumexp(logits, dim=-1),
        "target_k.1": torch.arange(32, dtype=torch.float32)
        .reshape(4, 2, 4)
        .to(torch.bfloat16),
        "target_v.1": (torch.arange(32, dtype=torch.float32) + 100)
        .reshape(4, 2, 4)
        .to(torch.bfloat16),
    }
    objects = []
    payloads = {}
    for index, name in enumerate(
        (
            "token_ids",
            "position_ids",
            "loss_mask",
            "kv_valid",
            "logits_positions",
            "teacher_topk_ids",
            "teacher_topk_logits",
            "teacher_logsumexp",
        )
    ):
        tensor = tensors[name]
        key = prefix + "aux/" + name
        data = _tensor_bytes(tensor)
        payloads[key] = data
        objects.append(
            {
                "object_id": f"aux-{index}",
                "name": name,
                "kind": "aux",
                "key": key,
                "dtype": str(tensor.dtype).removeprefix("torch."),
                "shape": list(tensor.shape),
                "nbytes": len(data),
                "sha256": _sha256(data),
                "owner_id": "dp0-pp0-tp0",
                "byte_order": "little",
                "contiguous": True,
            }
        )
    for component in ("k", "v"):
        name = f"target_{component}.1"
        tensor = tensors[name]
        key = prefix + f"kv/1/dp0-pp0-tp0/0/{component}"
        data = _tensor_bytes(tensor)
        payloads[key] = data
        objects.append(
            {
                "object_id": f"kv-1-{component}-0",
                "name": name,
                "kind": "kv",
                "key": key,
                "dtype": "bfloat16",
                "shape": list(tensor.shape),
                "nbytes": len(data),
                "sha256": _sha256(data),
                "owner_id": "dp0-pp0-tp0",
                "byte_order": "little",
                "contiguous": True,
                "layer_id": 1,
                "component": component,
                "token_range": [0, 4],
                "head_range": [0, 2],
            }
        )
    manifest = {
        "schema_version": 1,
        "payload_format": "maas_target_kv_v1",
        "contract_id": "maas-target-kv-top128-v1",
        "state": "READY",
        "input_mode": "target_kv",
        "dataset_id": "test-dataset",
        "sample_id": "test-sample",
        "generation_id": "g1",
        "created_at": "2026-10-08T00:00:00Z",
        "teacher": {
            "model_id": "teacher",
            "weights_revision": "weights-1",
            "adapter_revision": None,
            "tokenizer_revision": "tokenizer-1",
            "fingerprint_sha256": "1" * 64,
            "vocab_size": 256,
            "output_transform": "identity",
        },
        "sequence": {
            "prompt_length": prompt_length,
            "response_length": response_length,
            "total_length": total_length,
            "stop_reason": "length",
            "loss_mask_policy": "current_response_include_eos",
            "stop_token_policy": "preserve_internal_accepted_tokens",
            "position_ids_semantics": "actual_target_positions",
        },
        "kv": {
            "codec": "dense_bf16_post_rope_v1",
            "dtype": "bfloat16",
            "selected_layer_ids": [1],
            "layers": [
                {
                    "layer_id": 1,
                    "num_kv_heads": 2,
                    "key_head_dim": 4,
                    "value_head_dim": 4,
                }
            ],
            "source_k_stage": "post_rope",
            "source_k_norm": "none",
            "rope_config": rope,
            "rope_config_sha256": _sha256(
                json.dumps(rope, sort_keys=True, separators=(",", ":")).encode()
            ),
            "storage_chunk_tokens": 4,
            "source_page_size": 4,
            "layout": "token_head_dim",
            "layer_numbering": "target_attention_layer_zero_based",
            "validity_policy": "all_selected_layers_per_valid_token",
        },
        "logits": {
            "top_k": 128,
            "dtype": "float32",
            "semantics": "model_output_before_serving_processors",
            "normalization": "full_vocabulary_logsumexp",
            "lse_temperature": 1.0,
            "row_alignment": "predicts_token_at_logits_position",
            "vocab_ids": "global_unpadded",
        },
        "topology": {
            "tp_size": 1,
            "pp_size": 1,
            "aux_owner": "dp0-pp0-tp0",
            "owners": ["dp0-pp0-tp0"],
        },
        "provenance": {
            "capture_mode": "autoregressive",
            "producer_revision": "test",
            "capture_config_sha256": "2" * 64,
            "sampling_config": {},
            "trace_id": "trace-1",
        },
        "objects": objects,
        "total_tensor_bytes": sum(obj["nbytes"] for obj in objects),
        "extensions": {},
    }
    manifest_bytes = json.dumps(
        manifest, sort_keys=True, separators=(",", ":")
    ).encode()
    manifest_key = prefix + "manifest"
    payloads[manifest_key] = manifest_bytes
    publication = {
        "dataset_id": "test-dataset",
        "sample_id": "test-sample",
        "generation_id": "g1",
        "manifest_key": manifest_key,
        "manifest_sha256": _sha256(manifest_bytes),
        "manifest_nbytes": len(manifest_bytes),
        "contract_id": "maas-target-kv-top128-v1",
    }
    contract = {
        "input_mode": "target_kv",
        "teacher": manifest["teacher"],
        "kv": manifest["kv"],
    }
    return payloads, publication, tensors, contract


class _FakeMooncakeStore:
    def __init__(self, payloads):
        self.payloads = payloads

    def is_exist(self, key):
        return int(key in self.payloads)

    def register_buffer(self, pointer, size):
        return 0

    def unregister_buffer(self, pointer):
        return 0

    def get_into(self, key, pointer, size):
        data = self.payloads[key]
        ctypes.memmove(pointer, data, min(size, len(data)))
        return min(size, len(data))


class _UnregisterFailureStore(_FakeMooncakeStore):
    def unregister_buffer(self, pointer):
        return -1


class TestTrainingSnapshot(unittest.TestCase):
    def test_failed_unregister_keeps_backing_buffer_alive(self):
        payloads, publication, _source, contract = _snapshot()
        store = MooncakeSnapshotFeatureStore(
            store=_UnregisterFailureStore(payloads),
            expected_contract=contract,
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "publications.jsonl"
            path.write_text(json.dumps(publication) + "\n")
            ref = SnapshotRefReader(str(path), run_id="run-1").read()[0]
        with self.assertRaisesRegex(Exception, "unregister_buffer failed"):
            store.get(ref)
        self.assertEqual(store.health()["quarantined_buffers"], 1)

    def test_publication_to_validated_training_batch(self):
        payloads, publication, source, contract = _snapshot()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "publications.jsonl"
            path.write_text(json.dumps(publication) + "\n")
            ref = SnapshotRefReader(str(path), run_id="run-1").read()[0]

        store = MooncakeSnapshotFeatureStore(
            store=_FakeMooncakeStore(payloads),
            expected_contract=contract,
        )
        raw, handle = store.get(ref)
        self.assertTrue(torch.equal(raw["target_k.1"][:4], source["target_k.1"]))
        self.assertTrue(torch.equal(raw["target_k.1"][4], torch.zeros(2, 4)))
        normalized = normalize_target_kv_sample(raw, contract=contract, max_len=5)
        self.assertEqual(normalized["teacher_valid"].tolist(), [[0, 0, 1, 1, 1]])
        batch = build_target_kv_collator(contract)([normalized, normalized])
        self.assertEqual(batch["target_v.1"].shape, (2, 5, 2, 4))
        store.release(handle)
        store.close()

    def test_payload_corruption_is_rejected(self):
        payloads, publication, _, _ = _snapshot()
        key = next(key for key in payloads if key.endswith("teacher_topk_logits"))
        payloads[key] = bytes([payloads[key][0] ^ 0xFF]) + payloads[key][1:]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "publications.json"
            path.write_text(json.dumps([publication]))
            ref = SnapshotRefReader(str(path), run_id="run-1").read()[0]
        store = MooncakeSnapshotFeatureStore(store=_FakeMooncakeStore(payloads))
        with self.assertRaisesRegex(SnapshotContractError, "checksum"):
            store.get(ref)


if __name__ == "__main__":
    unittest.main()
