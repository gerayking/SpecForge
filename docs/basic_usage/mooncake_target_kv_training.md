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
