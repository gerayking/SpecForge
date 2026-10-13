# Train DSpark directly from Mooncake target-KV snapshots

This path consumes immutable `maas_target_kv_v1` samples written by a paired
SGLang capture deployment. SpecForge reads the manifest and tensor objects from
Mooncake, verifies their identities, sizes, SHA-256 digests, topology, tensor
contents, and draft contract, then trains DSpark without running target-model
prefill or recomputing teacher logits.

Each sample contains the full token and position sequence, the response loss
mask, selected target K/V layers, and one teacher row for every accepted
response token. A teacher row contains the raw top-128 vocabulary IDs and
logits plus the full-vocabulary `logsumexp`. The loss is
`CE + lambda_tv * TV128`; `lambda_tv` and the omitted-tail policy come from the
versioned target-KV contract in the draft checkpoint.

## Inputs

Use the `DSparkTargetKVDraftModel` checkpoint exported by the corresponding
SGLang build. The checkpoint binds the target identity, selected K/V geometry,
RoPE conversion, encoder shape, prediction window, vocabulary, output-logit
transform, and objective. SpecForge still loads the target embedding and LM
head as frozen shared modules. It does not load or execute the target decoder.

Export Catalog publications or claims as JSON, a JSON list, or JSONL. The file
contains metadata only; tensor bytes are fetched directly from Mooncake. Each
record has this shape:

```json
{
  "dataset_id": "maas-prod-2026-10-08",
  "sample_id": "01J...",
  "generation_id": "01J...",
  "manifest_key": "draft-data/maas-prod-2026-10-08/01J.../01J.../manifest",
  "manifest_sha256": "64 lowercase hexadecimal characters",
  "manifest_nbytes": 8192,
  "contract_id": "maas-target-kv-top128-v1"
}
```

The Catalog must retain every referenced manifest and tensor object for the
entire training run. SpecForge treats this source as read-only and never removes
remote objects.

## Run

Start from
`examples/configs/qwen3-4b-dspark-mooncake-target-kv.yaml`, set the checkpoint,
publication file, Mooncake endpoint, local address, and RDMA device, then run:

```bash
specforge train \
  --config examples/configs/qwen3-4b-dspark-mooncake-target-kv.yaml
```

Set `protocol: tcp` and remove `rdma_devices` for a TCP deployment. The reader
registers bounded CPU receive buffers with Mooncake, reads each object into its
final tensor allocation, validates it, and transfers the assembled batch to the
trainer device.

Snapshot training currently uses the trainer-only `local_colocated` topology.
Trainer TP/SP, evaluation, loss decay, and gradient accumulation are disabled;
data-parallel ranks and batch sizes greater than one remain supported. The
publication set is fixed at startup. Continuous Catalog claim/lease/ACK and
checkpoint-driven garbage collection require a future Catalog consumer API;
until then, create a stable publication export and pin its objects for the run.

## Train from a durable archive

For a capture that has been persisted as `index.json`, per-sample
`manifest.json`, and `tensors.safetensors`, replace the live endpoint fields
with one path:

```yaml
data:
  max_length: 3072
  mooncake_snapshots:
    archive_path: /datasets/qwen35-capture/snapshot-archive
    max_receive_bytes: 2147483648
```

`archive_path` and `refs_path` are mutually exclusive. The archive reader
validates the index file digests, every manifest, every tensor descriptor, and
every tensor checksum before a sample enters the training batch. It uses the
same target-KV model contract and `CE + lambda_tv * TV128` objective as the
live Mooncake reader.

## Experimental final-hidden snapshots

The reader also accepts `experimental_target_kv_final_hidden_v1` with contract
`experimental-target-kv-final-hidden-v1`, emitted by the Qwen3.5-4B capture
experiment. This format contains BF16 `teacher_final_hidden[R,2560]` instead of
top-128/LSE tensors. The hidden must be the **post-final-norm LM-head input**;
row `r` predicts the token at `logits_positions[r]`, including the first response
token. Pre-norm hidden and mixed hidden/top-k payloads are rejected.

Use the same `data.mooncake_snapshots` configuration for live Store refs or
archives. Live publication records must carry the hidden contract ID. The draft
contract must match the captured teacher identity and selected KV geometry;
the three-layer experiment requires layers **3, 15, 31**, rather than the
eight-layer example above. A source must use one teacher representation.

The normalizer aligns hidden rows to target token positions and preserves
`teacher_valid` through truncation and padding. During training, the frozen
target LM head processes only valid rows in chunks of 32. The configured target
output transform is applied, then raw top-128 IDs/logits and full-vocabulary
FP32 logsumexp are reconstructed under `no_grad`. The existing CE+TV128 objective
then runs unchanged; TV does not renormalize the retained top-128 mass. The
target transformer is neither loaded nor executed by this snapshot path.

This moves teacher-logit work to the trainer; it does not eliminate that work.
Stored compact-logit samples still use their original tensors. The checkpoint
records the actual snapshot format and rejects a different format on resume.
This experimental format remains restricted to width 2,560 and BF16.

A reproducible single-H100 integration smoke uses genuine serving archives,
replays their bytes into an isolated TCP Mooncake Store, then runs the standard
SpecForge trainer for two steps and resumes for two more:

```bash
python scripts/experiments/smoke_hidden_mooncake.py \
  --archives /data/hidden-readback-0 /data/hidden-readback-1 \
  --output /tmp/hidden-training-run
```

It requires the paired local Qwen3.5-4B weights, the `mooncake_master` binary,
the Mooncake Python SDK and SpecForge's normal training dependencies. Logs,
resolved recipes, read/reconstruction counters and checkpoints are written to
the output directory. This smoke verifies a fixed sample set over live Store
transport; concurrent serving, streaming claim/ACK, multi-GPU training and
trained-draft acceptance are separate validation steps.
