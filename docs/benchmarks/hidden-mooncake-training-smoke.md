# Final-hidden + KV capture to DSpark training

Date: 2026-10-13. A single-H100 integration smoke passed using real Qwen3.5-4B
serving captures with KV layers 3/15/31, final hidden, tokens and masks.

## Verified Path

The source consists of 16 distinct archived samples from the serving hidden
capture experiment. Each has 32 prompt and 32 response tokens. The driver first
validates the original manifests, file checksums and tensors, then republishes
the exact object bytes and manifests into an isolated live TCP Mooncake Store.
The normal SpecForge CLI reads publications from a fixed JSON file; all training
tensors are fetched by `MooncakeSnapshotFeatureStore` through the Store's
`get_into` API.

The reader accepts the explicit experimental hidden contract. Normalization
aligns each hidden row to its predicted response-token position. The training
model uses the frozen target LM head and the target output transform to compute
top-128 IDs/raw scores and full-vocabulary FP32 LSE in chunks of 32 valid rows.
No target transformer decoder or target prefill runs. The existing target-KV
DSpark objective remains CE + TV128 with lambda 1 and omitted-tail semantics.

Training uses the five-layer DSpark example architecture, BF16, SDPA, batch 1,
two anchors, and block size 16. It runs through the standard trainer and optimizer,
saves at step 2, exits, and starts a fresh process that resumes through step 4.
The GPU is an H100 80 GB on `node025`, pod
`job-83e3834bcf07-20261013140313-master-0`.

| Step | CE | TV128 | Total Loss | Gradient Norm |
| --- | ---: | ---: | ---: | ---: |
| 1 | 13.338032 | 0.484231 | 13.822262 | 252.019577 |
| 2 | 19.907837 | 0.512808 | 20.420645 | 78.174171 |
| 3, resumed | 14.531030 | 0.496519 | 15.027548 | 31.112339 |
| 4, resumed | 3.522529 | 0.782628 | 4.305157 | 6.627494 |

All losses and gradient norms are finite. These are different minibatches in a
four-step smoke, not evidence of convergence or useful speculative acceptance.

- Four distinct samples were fetched during training, two before and two after
  resume; there were 52 successful Store reads totaling 3,836,196 bytes.
- All 128 response hidden rows were reconstructed with no teacher autograd graph.
- Comparing step-2 and step-4 weights confirms updates in KV encoder, draft
  layers and Markov head. Checkpoints contain optimizer state; the restored
  sampler advances from two consumed samples to four.
- The snapshot builder now forwards its provider-bound `StepRuntimeConfig` to
  the trainer. Its previous omission blocked resume even though training worked.
- The checkpoint records `experimental_target_kv_final_hidden_v1`, so resuming
  with a different representation is rejected by the existing resume checks.

## Tests and Environment

Focused snapshot, objective, configuration, checkpoint/resume and teacher-strategy
regressions: **72 passed, 65 subtests passed, 3 skipped** with CUDA hidden from
that test process. The earlier GPU-enabled snapshot/objective run passed all
13 tests, including the CUDA forward/backward component-gradient test. New
coverage checks hidden alignment/truncation/padding, incompatible semantics,
corrupt payload rejection, mixed-format rejection, transformed teacher logits
and full-vocabulary LSE, frozen-head enforcement, and resume-contract forwarding.
Scoped Ruff F/E9 checks and `git diff --check` pass. The repository's broader
Ruff rules still report existing style issues; they were not globally reformatted.

The experiment uses the existing capture environment (PyTorch 2.13.0+cu130).
Yunchang 0.6.4 was installed in a separate `/tmp/hidden-train-deps` directory,
without modifying the shared environment. It does not use a Yunchang shim.
The run sets `HF_HUB_OFFLINE=1`, `OMP_NUM_THREADS=1`, and
`FLASHINFER_DISABLE_VERSION_CHECK=1` for the existing image's FlashInfer mismatch.
The smoke driver removes inherited cluster rank variables from its single-rank
worker environment. Serving and training did not run concurrently.

## Artifacts and Limits

Local artifacts are retained at:

`/gpfs/user/fuxuanwei/sglang/mooncake-study/experiments/hidden-training-20261013`

They include `report.json`, `checkpoint-validation.json`, resolved draft and
training configs, publication refs, training/resume logs, the training-source
archive, and the complete final `checkpoint-step4/` (9,065,490,780 bytes).
The final checkpoint is preserved outside the Git repository; both checkpoint
files have matching SHA-256 digests before and after copying from the pod.
Original runtime
paths in the recipes/report refer to `/tmp/hidden-training-run3` on the pod;
the published test Store has been closed, so reruns must republish the samples.
Use `scripts/experiments/smoke_hidden_mooncake.py` as documented in the
[training guide](../basic_usage/mooncake_target_kv_training.md).

This verifies training from a fixed set of genuine captures over a live Store,
not simultaneous serving/streaming training. Continuous Catalog claim/lease/ACK,
multi-GPU data consumption, cross-node RDMA, draft serving parity and useful
trained-model acceptance remain unverified. The generated draft config starts
from the existing training template; its fixture metadata is not a new serving
parity certification. No further serving-performance claim follows from this run.
