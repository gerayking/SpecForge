# coding=utf-8
"""Read SGLang ``maas_target_kv_v1`` snapshots directly from Mooncake."""

from __future__ import annotations

import hashlib
import json
import math
import re
import sys
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Tuple

import torch

from specforge.runtime.contracts import FeatureHandle, SampleRef
from specforge.runtime.data_plane.feature_store import FeatureStore

PAYLOAD_FORMAT = "maas_target_kv_v1"
CONTRACT_ID = "maas-target-kv-top128-v1"
_MAX_MANIFEST_BYTES = 8 << 20
_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,159}$")
_DIGEST = re.compile(r"^[a-f0-9]{64}$")
_DTYPES = {
    "int32": torch.int32,
    "int64": torch.int64,
    "uint8": torch.uint8,
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}
_ELEMENT_BYTES = {
    name: torch.empty((), dtype=dtype).element_size() for name, dtype in _DTYPES.items()
}
_AUX_NAMES = {
    "token_ids",
    "position_ids",
    "loss_mask",
    "kv_valid",
    "logits_positions",
    "teacher_topk_ids",
    "teacher_topk_logits",
    "teacher_logsumexp",
}


class SnapshotContractError(RuntimeError):
    pass


class SnapshotTransportError(RuntimeError):
    pass


def _sha256(data: bytes | memoryview) -> str:
    return hashlib.sha256(data).hexdigest()


def _tensor_bytes(tensor: torch.Tensor) -> memoryview:
    if (
        sys.byteorder != "little"
        or tensor.device.type != "cpu"
        or not tensor.is_contiguous()
    ):
        raise SnapshotContractError(
            "snapshot transport requires little-endian contiguous CPU tensors"
        )
    return memoryview(tensor.detach().view(torch.uint8).numpy()).cast("B")


def _require_dict(value: Any, name: str) -> dict:
    if not isinstance(value, dict):
        raise SnapshotContractError(f"{name} must be an object")
    return value


def _require_identifier(value: Any, name: str) -> str:
    if not isinstance(value, str) or _IDENTIFIER.fullmatch(value) is None:
        raise SnapshotContractError(f"invalid {name}")
    return value


def _require_digest(value: Any, name: str) -> str:
    if not isinstance(value, str) or _DIGEST.fullmatch(value) is None:
        raise SnapshotContractError(f"invalid {name}")
    return value


def _positive_int(value: Any, name: str, *, allow_zero: bool = False) -> int:
    minimum = 0 if allow_zero else 1
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise SnapshotContractError(f"{name} must be an integer >= {minimum}")
    return value


def _range(value: Any, name: str, limit: int) -> Tuple[int, int]:
    if not isinstance(value, list) or len(value) != 2:
        raise SnapshotContractError(f"{name} must be a two-element range")
    start = _positive_int(value[0], f"{name}[0]", allow_zero=True)
    stop = _positive_int(value[1], f"{name}[1]")
    if not start < stop <= limit:
        raise SnapshotContractError(f"{name} is out of bounds")
    return start, stop


def _aux_specs(
    total_length: int, response_length: int
) -> Dict[str, Tuple[str, list[int]]]:
    return {
        "token_ids": ("int32", [total_length]),
        "position_ids": ("int64", [total_length]),
        "loss_mask": ("uint8", [total_length]),
        "kv_valid": ("uint8", [total_length]),
        "logits_positions": ("int32", [response_length]),
        "teacher_topk_ids": ("int32", [response_length, 128]),
        "teacher_topk_logits": ("float32", [response_length, 128]),
        "teacher_logsumexp": ("float32", [response_length]),
    }


def _validate_kv_coverage(objects: list[dict], total_length: int, heads: int) -> int:
    if not objects:
        raise SnapshotContractError("missing selected KV layer/component")
    coverage_end = max(obj["token_range"][1] for obj in objects)
    if coverage_end not in (total_length - 1, total_length):
        raise SnapshotContractError("only the final token may lack KV")
    boundaries = sorted(
        {0, coverage_end} | {point for obj in objects for point in obj["token_range"]}
    )
    for token_start, token_stop in zip(boundaries, boundaries[1:]):
        intervals = sorted(
            obj["head_range"]
            for obj in objects
            if obj["token_range"][0] <= token_start
            and obj["token_range"][1] >= token_stop
        )
        cursor = 0
        for head_start, head_stop in intervals:
            if head_start != cursor:
                raise SnapshotContractError("overlapping or missing KV coverage")
            cursor = head_stop
        if cursor != heads:
            raise SnapshotContractError("incomplete KV head coverage")
    return coverage_end


def validate_manifest(
    manifest: Mapping[str, Any], *, max_tensor_bytes: int = 2 << 30
) -> int:
    """Validate all allocation-driving metadata and return the valid KV length."""

    root = _require_dict(manifest, "manifest")
    required_root = {
        "schema_version",
        "payload_format",
        "contract_id",
        "state",
        "input_mode",
        "dataset_id",
        "sample_id",
        "generation_id",
        "created_at",
        "teacher",
        "sequence",
        "kv",
        "logits",
        "topology",
        "provenance",
        "objects",
        "total_tensor_bytes",
    }
    allowed_root = required_root | {"extensions"}
    if set(root) - allowed_root or not required_root.issubset(root):
        raise SnapshotContractError("manifest root fields do not match schema v1")
    if (
        root["schema_version"] != 1
        or root["payload_format"] != PAYLOAD_FORMAT
        or root["contract_id"] != CONTRACT_ID
        or root["state"] != "READY"
        or root["input_mode"] != "target_kv"
    ):
        raise SnapshotContractError("unsupported snapshot contract")
    for field in ("dataset_id", "sample_id", "generation_id"):
        _require_identifier(root[field], field)
    try:
        created = datetime.fromisoformat(str(root["created_at"]).replace("Z", "+00:00"))
        if created.tzinfo is None:
            raise ValueError
    except ValueError as exc:
        raise SnapshotContractError("created_at must be an RFC3339 timestamp") from exc
    if _require_dict(root.get("extensions", {}), "extensions").get("example_only"):
        raise SnapshotContractError("example manifests are not training samples")

    sequence = _require_dict(root["sequence"], "sequence")
    prompt_length = _positive_int(sequence.get("prompt_length"), "prompt_length")
    response_length = _positive_int(sequence.get("response_length"), "response_length")
    total_length = _positive_int(sequence.get("total_length"), "total_length")
    if total_length != prompt_length + response_length:
        raise SnapshotContractError("total length must equal prompt plus response")
    if sequence.get("position_ids_semantics") != "actual_target_positions":
        raise SnapshotContractError("unsupported position semantics")

    teacher = _require_dict(root["teacher"], "teacher")
    vocab_size = _positive_int(teacher.get("vocab_size"), "teacher.vocab_size")
    if vocab_size < 128:
        raise SnapshotContractError("teacher vocabulary is smaller than top-128")
    _require_digest(teacher.get("fingerprint_sha256"), "teacher fingerprint")

    logits = _require_dict(root["logits"], "logits")
    expected_logits = {
        "top_k": 128,
        "dtype": "float32",
        "semantics": "model_output_before_serving_processors",
        "normalization": "full_vocabulary_logsumexp",
        "lse_temperature": 1.0,
        "row_alignment": "predicts_token_at_logits_position",
        "vocab_ids": "global_unpadded",
    }
    if any(logits.get(key) != value for key, value in expected_logits.items()):
        raise SnapshotContractError("unsupported teacher logits semantics")

    kv = _require_dict(root["kv"], "kv")
    dtype = kv.get("dtype")
    source_k_stage = kv.get("source_k_stage")
    codec_prefix = (
        "bf16" if dtype == "bfloat16" else "fp16" if dtype == "float16" else None
    )
    if (
        codec_prefix is None
        or kv.get("codec") != f"dense_{codec_prefix}_{source_k_stage}_v1"
    ):
        raise SnapshotContractError("KV codec disagrees with dtype or K stage")
    if (
        kv.get("layout") != "token_head_dim"
        or kv.get("validity_policy") != "all_selected_layers_per_valid_token"
    ):
        raise SnapshotContractError("unsupported KV layout or validity policy")
    selected_layers = kv.get("selected_layer_ids")
    layers = kv.get("layers")
    if (
        not isinstance(selected_layers, list)
        or not selected_layers
        or len(set(selected_layers)) != len(selected_layers)
        or not isinstance(layers, list)
        or not all(isinstance(layer, dict) for layer in layers)
        or [layer.get("layer_id") for layer in layers] != selected_layers
    ):
        raise SnapshotContractError("invalid selected KV layer geometry")
    rope_config = _require_dict(kv.get("rope_config"), "kv.rope_config")
    rope_bytes = json.dumps(
        rope_config, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    if _sha256(rope_bytes) != _require_digest(
        kv.get("rope_config_sha256"), "rope config digest"
    ):
        raise SnapshotContractError("RoPE configuration digest mismatch")

    topology = _require_dict(root["topology"], "topology")
    owners = topology.get("owners")
    aux_owner = topology.get("aux_owner")
    if (
        not isinstance(owners, list)
        or not owners
        or len(set(owners)) != len(owners)
        or aux_owner not in owners
    ):
        raise SnapshotContractError("invalid owner topology")
    for owner in owners:
        _require_identifier(owner, "owner")

    objects = root["objects"]
    total_tensor_bytes = _positive_int(root["total_tensor_bytes"], "total_tensor_bytes")
    if (
        not isinstance(objects, list)
        or len(objects) > 16384
        or total_tensor_bytes > max_tensor_bytes
    ):
        raise SnapshotContractError("sample exceeds the reader allocation budget")
    prefix = (
        f"draft-data/{root['dataset_id']}/{root['sample_id']}/{root['generation_id']}/"
    )
    aux_specs = _aux_specs(total_length, response_length)
    geometries = {layer["layer_id"]: layer for layer in layers}
    seen_ids: set[str] = set()
    seen_keys: set[str] = set()
    seen_aux: set[str] = set()
    groups: Dict[Tuple[int, str], list[dict]] = {}
    total = 0
    for raw_obj in objects:
        obj = _require_dict(raw_obj, "tensor descriptor")
        required = {
            "object_id",
            "name",
            "kind",
            "key",
            "dtype",
            "shape",
            "nbytes",
            "sha256",
            "owner_id",
            "byte_order",
            "contiguous",
        }
        if not required.issubset(obj):
            raise SnapshotContractError("tensor descriptor is missing required fields")
        object_id = _require_identifier(obj["object_id"], "object_id")
        key = obj["key"]
        if object_id in seen_ids or key in seen_keys:
            raise SnapshotContractError("duplicate snapshot object")
        seen_ids.add(object_id)
        seen_keys.add(key)
        if (
            not isinstance(key, str)
            or not key.startswith(prefix)
            or ".." in key.split("/")
            or obj["owner_id"] not in owners
            or obj["byte_order"] != "little"
            or obj["contiguous"] is not True
        ):
            raise SnapshotContractError("invalid object namespace, owner, or layout")
        shape = obj["shape"]
        object_dtype = obj["dtype"]
        if (
            object_dtype not in _DTYPES
            or not isinstance(shape, list)
            or not shape
            or any(
                isinstance(dim, bool) or not isinstance(dim, int) or dim < 1
                for dim in shape
            )
        ):
            raise SnapshotContractError("invalid tensor dtype or shape")
        expected_nbytes = math.prod(shape) * _ELEMENT_BYTES[object_dtype]
        if obj["nbytes"] != expected_nbytes:
            raise SnapshotContractError("tensor byte count disagrees with shape")
        _require_digest(obj["sha256"], "object digest")
        total += expected_nbytes
        if total > max_tensor_bytes:
            raise SnapshotContractError("tensor allocation budget exceeded")

        if obj["kind"] == "aux":
            name = obj["name"]
            if name not in _AUX_NAMES or name in seen_aux:
                raise SnapshotContractError("unknown or duplicate aux tensor")
            if (
                (object_dtype, shape) != aux_specs[name]
                or obj["owner_id"] != aux_owner
                or key != prefix + "aux/" + name
                or any(
                    field in obj
                    for field in (
                        "layer_id",
                        "component",
                        "token_range",
                        "head_range",
                    )
                )
            ):
                raise SnapshotContractError("aux tensor descriptor mismatch")
            seen_aux.add(name)
            continue
        if obj["kind"] != "kv":
            raise SnapshotContractError("unknown snapshot object kind")
        layer_id = obj.get("layer_id")
        component = obj.get("component")
        if layer_id not in geometries or component not in ("k", "v"):
            raise SnapshotContractError("invalid KV layer or component")
        geometry = geometries[layer_id]
        token_range = _range(obj.get("token_range"), "token_range", total_length)
        head_range = _range(
            obj.get("head_range"), "head_range", geometry["num_kv_heads"]
        )
        dimension = (
            geometry["key_head_dim"] if component == "k" else geometry["value_head_dim"]
        )
        if (
            shape
            != [
                token_range[1] - token_range[0],
                head_range[1] - head_range[0],
                dimension,
            ]
            or object_dtype != dtype
            or obj["name"] != f"target_{component}.{layer_id}"
        ):
            raise SnapshotContractError("KV tensor geometry mismatch")
        stem = prefix + f"kv/{layer_id}/{obj['owner_id']}/"
        suffix = key.removeprefix(stem).split("/")
        if (
            not key.startswith(stem)
            or len(suffix) != 2
            or not suffix[0].isdigit()
            or suffix[1] != component
        ):
            raise SnapshotContractError("noncanonical KV object key")
        groups.setdefault((layer_id, component), []).append(obj)

    if seen_aux != _AUX_NAMES or total != total_tensor_bytes:
        raise SnapshotContractError("missing aux fields or inconsistent total bytes")
    if {obj["owner_id"] for obj in objects} != set(owners):
        raise SnapshotContractError("manifest has no objects for one or more owners")
    coverage = {
        _validate_kv_coverage(
            groups.get((layer_id, component), []),
            total_length,
            geometries[layer_id]["num_kv_heads"],
        )
        for layer_id in selected_layers
        for component in ("k", "v")
    }
    if len(coverage) != 1:
        raise SnapshotContractError("selected layers have inconsistent KV validity")
    return coverage.pop()


def decode_manifest(
    data: bytes,
    *,
    max_manifest_bytes: int = _MAX_MANIFEST_BYTES,
    max_tensor_bytes: int = 2 << 30,
) -> dict:
    if not data or len(data) > max_manifest_bytes:
        raise SnapshotContractError("manifest exceeds metadata budget")
    try:
        manifest = json.loads(data)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SnapshotContractError("invalid snapshot manifest JSON") from exc
    validate_manifest(manifest, max_tensor_bytes=max_tensor_bytes)
    return manifest


def validate_tensors(
    manifest: Mapping[str, Any], tensors: Mapping[str, torch.Tensor]
) -> None:
    valid_length = validate_manifest(manifest)
    objects = manifest["objects"]
    if set(tensors) != {obj["key"] for obj in objects}:
        raise SnapshotContractError("tensor object set differs from manifest")
    aux: Dict[str, torch.Tensor] = {}
    for obj in objects:
        tensor = tensors[obj["key"]]
        if list(tensor.shape) != obj["shape"] or tensor.dtype != _DTYPES[obj["dtype"]]:
            raise SnapshotContractError("received tensor shape or dtype mismatch")
        if _sha256(_tensor_bytes(tensor)) != obj["sha256"]:
            raise SnapshotContractError("snapshot tensor checksum mismatch")
        if tensor.is_floating_point() and not torch.isfinite(tensor).all():
            raise SnapshotContractError("snapshot tensor contains nonfinite values")
        if obj["kind"] == "aux":
            aux[obj["name"]] = tensor
    sequence = manifest["sequence"]
    total_length = sequence["total_length"]
    prompt_length = sequence["prompt_length"]
    vocab_size = manifest["teacher"]["vocab_size"]
    if not ((aux["token_ids"] >= 0) & (aux["token_ids"] < vocab_size)).all():
        raise SnapshotContractError("token IDs are outside the teacher vocabulary")
    expected_positions = torch.arange(prompt_length, total_length, dtype=torch.int32)
    if not torch.equal(aux["logits_positions"], expected_positions):
        raise SnapshotContractError("teacher rows do not cover the accepted response")
    positions = aux["position_ids"]
    if not ((positions >= 0).all() and (positions[1:] > positions[:-1]).all()):
        raise SnapshotContractError("position IDs must be strictly increasing")
    loss_mask = aux["loss_mask"]
    if loss_mask[:prompt_length].any() or (loss_mask > 1).any():
        raise SnapshotContractError("invalid response loss mask")
    if (
        sequence["loss_mask_policy"] == "current_response_include_eos"
        and not loss_mask[prompt_length:].all()
    ):
        raise SnapshotContractError("response loss mask has missing supervision")
    expected_valid = torch.zeros(total_length, dtype=torch.uint8)
    expected_valid[:valid_length] = 1
    if not torch.equal(aux["kv_valid"], expected_valid):
        raise SnapshotContractError("KV validity differs from object coverage")
    ids = aux["teacher_topk_ids"]
    if not ((ids >= 0) & (ids < vocab_size)).all():
        raise SnapshotContractError("teacher top-k IDs are outside the vocabulary")
    sorted_ids = ids.sort(dim=-1).values
    if (sorted_ids[:, 1:] == sorted_ids[:, :-1]).any():
        raise SnapshotContractError("teacher top-k IDs contain duplicates")
    topk_logits = aux["teacher_topk_logits"]
    if (topk_logits[:, 1:] > topk_logits[:, :-1]).any():
        raise SnapshotContractError("teacher top-k logits are not sorted")
    topk_mass = torch.exp(topk_logits - aux["teacher_logsumexp"].unsqueeze(-1)).sum(-1)
    if (topk_mass > 1.00002).any():
        raise SnapshotContractError("teacher top-k mass exceeds full-vocabulary mass")


def _publication_records(path: str) -> Iterable[dict]:
    data = Path(path).read_bytes()
    if not data:
        return []
    try:
        decoded = json.loads(data)
    except json.JSONDecodeError:
        records = []
        for line_number, line in enumerate(data.splitlines(), 1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise SnapshotContractError(
                    f"invalid publication JSONL at line {line_number}"
                ) from exc
        return records
    if isinstance(decoded, list):
        return decoded
    if isinstance(decoded, dict):
        for key in ("publications", "claims", "refs"):
            if key in decoded:
                if not isinstance(decoded[key], list):
                    raise SnapshotContractError(
                        f"publication field {key!r} must be a list"
                    )
                return decoded[key]
        return [decoded]
    raise SnapshotContractError(
        "publication file must contain an object, list, or JSONL"
    )


class SnapshotRefReader:
    """Convert Catalog claim/publication metadata into tensor-free SampleRefs."""

    def __init__(self, path: str, *, run_id: str, strategy: str = "dspark") -> None:
        self.path = path
        self.run_id = run_id
        self.strategy = strategy

    def read(self) -> list[SampleRef]:
        refs = []
        identities = set()
        for raw in _publication_records(self.path):
            record = _require_dict(raw, "publication")
            required = {
                "dataset_id",
                "sample_id",
                "generation_id",
                "manifest_key",
                "manifest_sha256",
                "manifest_nbytes",
                "contract_id",
            }
            if not required.issubset(record):
                raise SnapshotContractError("publication is missing manifest identity")
            if record["contract_id"] != CONTRACT_ID:
                raise SnapshotContractError("publication uses an unsupported contract")
            dataset_id = _require_identifier(record["dataset_id"], "dataset_id")
            sample_id = _require_identifier(record["sample_id"], "sample_id")
            generation_id = _require_identifier(
                record["generation_id"], "generation_id"
            )
            identity = (dataset_id, sample_id, generation_id)
            if identity in identities:
                raise SnapshotContractError("duplicate publication identity")
            identities.add(identity)
            manifest_key = record["manifest_key"]
            expected_key = (
                f"draft-data/{dataset_id}/{sample_id}/{generation_id}/manifest"
            )
            if manifest_key != expected_key:
                raise SnapshotContractError("publication manifest key is noncanonical")
            manifest_nbytes = _positive_int(
                record["manifest_nbytes"], "manifest_nbytes"
            )
            if manifest_nbytes > _MAX_MANIFEST_BYTES:
                raise SnapshotContractError(
                    "publication manifest exceeds metadata budget"
                )
            manifest_digest = _require_digest(
                record["manifest_sha256"], "manifest digest"
            )
            refs.append(
                SampleRef(
                    sample_id=f"{dataset_id}:{sample_id}:{generation_id}",
                    run_id=self.run_id,
                    source_task_id=None,
                    feature_store_uri=f"mooncake-snapshot://{manifest_key}",
                    feature_keys={},
                    feature_specs={},
                    strategy=self.strategy,
                    target_model_version=str(
                        record.get("target_model_version", "manifest-bound")
                    ),
                    tokenizer_version=str(
                        record.get("tokenizer_version", "manifest-bound")
                    ),
                    num_tokens=int(record.get("num_tokens", 0)),
                    estimated_bytes=int(record.get("estimated_bytes", 0)),
                    metadata={
                        "payload_format": PAYLOAD_FORMAT,
                        "contract_id": CONTRACT_ID,
                        "dataset_id": dataset_id,
                        "manifest_sample_id": sample_id,
                        "generation_id": generation_id,
                        "manifest_key": manifest_key,
                        "manifest_sha256": manifest_digest,
                        "manifest_nbytes": manifest_nbytes,
                        "target_repr": "topk_logits_lse",
                        "read_lease": record.get("read_lease"),
                        "claim_token": record.get("claim_token"),
                    },
                )
            )
        if not refs:
            raise SnapshotContractError("publication source contains no samples")
        return refs


class MooncakeSnapshotFeatureStore(FeatureStore):
    """Manifest-aware read-only FeatureStore; Catalog remains retention owner."""

    def __init__(
        self,
        *,
        store: Optional[Any] = None,
        setup_kwargs: Optional[Dict[str, Any]] = None,
        max_receive_bytes: int = 2 << 30,
        expected_contract: Optional[Mapping[str, Any]] = None,
    ) -> None:
        if max_receive_bytes <= 0:
            raise ValueError("max_receive_bytes must be positive")
        self._owns_store = store is None
        if store is None:
            try:
                from mooncake.store import MooncakeDistributedStore
            except Exception as exc:
                raise RuntimeError(
                    "Mooncake snapshot training requires mooncake-transfer-engine"
                ) from exc
            store = MooncakeDistributedStore()
            rc = store.setup(**dict(setup_kwargs or {}))
            if rc not in (None, 0):
                raise SnapshotTransportError(f"Mooncake setup failed with status {rc}")
        required = ("is_exist", "get_into", "register_buffer", "unregister_buffer")
        missing = [
            name for name in required if not callable(getattr(store, name, None))
        ]
        if missing:
            raise RuntimeError(f"Mooncake store is missing raw-buffer APIs: {missing}")
        self._store = store
        self.max_receive_bytes = int(max_receive_bytes)
        self.expected_contract = dict(expected_contract or {})
        self._active: Dict[str, FeatureHandle] = {}
        self._quarantined: list[torch.Tensor] = []
        self._counter = 0
        self._lock = threading.RLock()
        self._closed = False

    def put(self, tensors, *, sample_id, metadata):
        raise TypeError("MooncakeSnapshotFeatureStore is read-only")

    def _get_raw(self, key: str, shape: list[int], dtype: torch.dtype) -> torch.Tensor:
        nbytes = math.prod(shape) * torch.empty((), dtype=dtype).element_size()
        if nbytes <= 0 or nbytes > self.max_receive_bytes:
            raise SnapshotContractError("object exceeds the receive budget")
        exists = self._store.is_exist(key)
        if int(exists) != 1:
            raise KeyError(f"Mooncake snapshot object is unavailable: {key}")
        tensor = torch.empty(shape, dtype=dtype)
        pointer = tensor.data_ptr()
        rc = self._store.register_buffer(pointer, nbytes)
        if rc not in (None, 0):
            raise SnapshotTransportError(f"Mooncake register_buffer failed: {rc}")
        uncertain = False
        try:
            count = self._store.get_into(key, pointer, nbytes)
            if isinstance(count, bool) or not isinstance(count, int) or count < 0:
                uncertain = True
                raise SnapshotTransportError(f"Mooncake get_into failed: {count}")
            if count != nbytes:
                raise SnapshotContractError(
                    f"short Mooncake read for {key}: expected {nbytes}, got {count}"
                )
        except Exception:
            uncertain = True
            # The SDK may still own the registered address after an interrupted
            # transfer. Keep its backing allocation alive until Store shutdown.
            self._quarantined.append(tensor)
            raise
        finally:
            if not uncertain:
                unregistered = self._store.unregister_buffer(pointer)
                if unregistered not in (None, 0):
                    self._quarantined.append(tensor)
                    raise SnapshotTransportError(
                        f"Mooncake unregister_buffer failed: {unregistered}"
                    )
        return tensor

    def _read_manifest(self, ref: SampleRef) -> dict:
        metadata = ref.metadata
        nbytes = _positive_int(metadata.get("manifest_nbytes"), "manifest_nbytes")
        raw = self._get_raw(metadata["manifest_key"], [nbytes], torch.uint8)
        data = bytes(_tensor_bytes(raw))
        if _sha256(data) != metadata["manifest_sha256"]:
            raise SnapshotContractError("manifest digest mismatch")
        manifest = decode_manifest(data, max_tensor_bytes=self.max_receive_bytes)
        identity = (
            manifest["dataset_id"],
            manifest["sample_id"],
            manifest["generation_id"],
        )
        expected = (
            metadata["dataset_id"],
            metadata["manifest_sample_id"],
            metadata["generation_id"],
        )
        if identity != expected:
            raise SnapshotContractError("publication and manifest identities differ")
        self._validate_expected_contract(manifest)
        return manifest

    def _validate_expected_contract(self, manifest: Mapping[str, Any]) -> None:
        contract = self.expected_contract
        if not contract:
            return
        if contract.get("input_mode") != "target_kv":
            raise SnapshotContractError("draft checkpoint is not a target-KV model")
        if contract.get("teacher") != manifest["teacher"]:
            raise SnapshotContractError("snapshot teacher differs from draft contract")
        expected_kv = contract.get("kv")
        if not isinstance(expected_kv, dict):
            raise SnapshotContractError("draft target-KV contract has no KV spec")
        semantic_fields = (
            "codec",
            "dtype",
            "selected_layer_ids",
            "layers",
            "source_k_stage",
            "source_k_norm",
            "rope_config",
            "rope_config_sha256",
            "layout",
            "layer_numbering",
            "validity_policy",
        )
        mismatches = [
            field
            for field in semantic_fields
            if expected_kv.get(field) != manifest["kv"].get(field)
        ]
        if mismatches:
            raise SnapshotContractError(
                f"snapshot KV differs from draft contract: {mismatches}"
            )

    @staticmethod
    def _materialize_features(
        manifest: Mapping[str, Any], objects: Mapping[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        features: Dict[str, torch.Tensor] = {}
        total_length = manifest["sequence"]["total_length"]
        geometries = {layer["layer_id"]: layer for layer in manifest["kv"]["layers"]}
        for descriptor in manifest["objects"]:
            if descriptor["kind"] == "aux":
                features[descriptor["name"]] = objects[descriptor["key"]]
        for layer_id in manifest["kv"]["selected_layer_ids"]:
            geometry = geometries[layer_id]
            for component in ("k", "v"):
                dimension = (
                    geometry["key_head_dim"]
                    if component == "k"
                    else geometry["value_head_dim"]
                )
                output = torch.zeros(
                    total_length,
                    geometry["num_kv_heads"],
                    dimension,
                    dtype=_DTYPES[manifest["kv"]["dtype"]],
                )
                for descriptor in manifest["objects"]:
                    if (
                        descriptor["kind"] == "kv"
                        and descriptor["layer_id"] == layer_id
                        and descriptor["component"] == component
                    ):
                        token_start, token_stop = descriptor["token_range"]
                        head_start, head_stop = descriptor["head_range"]
                        output[token_start:token_stop, head_start:head_stop].copy_(
                            objects[descriptor["key"]]
                        )
                features[f"target_{component}.{layer_id}"] = output
        return features

    def get(
        self,
        sample_ref: SampleRef,
        *,
        device: "torch.device | str" = "cpu",
        names: Optional[list[str]] = None,
    ) -> Tuple[Dict[str, torch.Tensor], FeatureHandle]:
        if names is not None:
            raise ValueError("snapshot reads materialize one complete validated sample")
        with self._lock:
            if self._closed:
                raise RuntimeError("MooncakeSnapshotFeatureStore is closed")
            manifest = self._read_manifest(sample_ref)
            objects = {
                descriptor["key"]: self._get_raw(
                    descriptor["key"], descriptor["shape"], _DTYPES[descriptor["dtype"]]
                )
                for descriptor in manifest["objects"]
            }
            validate_tensors(manifest, objects)
            tensors = self._materialize_features(manifest, objects)
            if str(device) != "cpu":
                tensors = {name: tensor.to(device) for name, tensor in tensors.items()}
            self._counter += 1
            handle = FeatureHandle(
                sample_id=sample_ref.sample_id,
                generation=self._counter,
                lease_token=f"snapshot:{self._counter}:{sample_ref.sample_id}",
            )
            self._active[handle.lease_token] = handle
            return tensors, handle

    def release(self, handle: FeatureHandle, *, reason: str = "consumed") -> None:
        del reason
        with self._lock:
            self._active.pop(handle.lease_token, None)

    def abort(self, sample_id: str, *, reason: str = "aborted") -> None:
        del reason
        with self._lock:
            for token, handle in list(self._active.items()):
                if handle.sample_id == sample_id:
                    self._active.pop(token, None)

    def gc(self) -> dict:
        return {"force_freed": 0, "force_freed_bytes": 0}

    def health(self) -> dict:
        with self._lock:
            return {
                "active_leases": len(self._active),
                "quarantined_buffers": len(self._quarantined),
                "retention_owner": "catalog",
                "read_only": True,
            }

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            if self._active:
                raise RuntimeError(
                    "cannot close snapshot store with active read handles"
                )
            if self._owns_store:
                rc = self._store.close()
                if rc not in (None, 0):
                    raise SnapshotTransportError(f"Mooncake close failed: {rc}")
                self._quarantined.clear()
            self._closed = True


__all__ = [
    "CONTRACT_ID",
    "PAYLOAD_FORMAT",
    "MooncakeSnapshotFeatureStore",
    "SnapshotContractError",
    "SnapshotRefReader",
    "SnapshotTransportError",
    "decode_manifest",
    "validate_manifest",
    "validate_tensors",
]
