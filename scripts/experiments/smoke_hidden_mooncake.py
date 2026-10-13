"""Train/resume DSpark through live Mooncake using archived serving captures.

The archive is replayed byte-for-byte into an isolated Store. This verifies the
real transport and trainer, not a continuously running serving/claim pipeline.
"""

import argparse
import hashlib
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + "\n")


def free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def verify_checkpoints(root):
    import torch

    root = Path(root)
    states = [
        torch.load(
            root / f"checkpoints/hidden-mooncake-smoke-step{step}/training_state.pt",
            map_location="cpu",
            weights_only=False,
            mmap=True,
        )
        for step in (2, 4)
    ]
    before, after = [state["draft_state_dict"] for state in states]
    changed = {}
    for group in ("kv_encoder", "layers", "markov_head"):
        keys = [key for key in after if key.startswith(group + ".")]
        changed[group] = any(not torch.equal(before[key], after[key]) for key in keys)
    assert all(changed.values()), changed
    assert all(state["replicated_optimizer_state"] for state in states)
    assert [state["epoch_samples"] for state in states] == [2, 4]
    return {
        "parameter_groups_changed_after_resume": changed,
        "optimizer_state_present": True,
        "epoch_samples": [2, 4],
    }


def worker(config, evidence_path):
    import torch

    from specforge import cli
    from specforge.algorithms.common.target_kv_model import OnlineTargetKVDSparkModel
    from specforge.runtime.data_plane.training_snapshot import (
        MooncakeSnapshotFeatureStore,
    )
    from specforge.training import assembly

    evidence = {
        "status": "running",
        "metrics": [],
        "samples_read": [],
        "teacher_rows": 0,
        "read_calls": 0,
        "read_bytes": 0,
        "teacher_has_grad": False,
    }
    original_get = MooncakeSnapshotFeatureStore.get
    original_raw = MooncakeSnapshotFeatureStore._get_raw
    original_teacher = OnlineTargetKVDSparkModel.teacher_from_hidden
    original_logger = assembly._logger

    def get(self, ref, **kwargs):
        result = original_get(self, ref, **kwargs)
        evidence["samples_read"].append(ref.sample_id)
        return result

    def raw(self, *args, **kwargs):
        value = original_raw(self, *args, **kwargs)
        evidence["read_calls"] += 1
        evidence["read_bytes"] += value.numel() * value.element_size()
        return value

    def teacher(self, hidden, valid, **kwargs):
        values = original_teacher(self, hidden, valid, **kwargs)
        evidence["teacher_rows"] += int(valid.sum())
        evidence["teacher_has_grad"] |= any(value.requires_grad for value in values)
        return values

    def logger(metrics, step):
        original_logger(metrics, step)
        evidence["metrics"].append(
            {
                "step": step,
                **{
                    key: float(value)
                    for key, value in metrics.items()
                    if isinstance(value, (int, float))
                    or (isinstance(value, torch.Tensor) and value.numel() == 1)
                },
            }
        )

    MooncakeSnapshotFeatureStore.get = get
    MooncakeSnapshotFeatureStore._get_raw = raw
    OnlineTargetKVDSparkModel.teacher_from_hidden = teacher
    assembly._logger = logger
    try:
        rc = cli.main(["train", "--config", str(config)])
        if rc:
            raise RuntimeError(f"trainer exited {rc}")
        evidence["status"] = "passed"
    finally:
        write_json(evidence_path, evidence)


def run(args):
    import torch
    import yaml
    from mooncake.store import MooncakeDistributedStore
    from safetensors.torch import load_file

    from specforge.runtime.data_plane.training_snapshot import (
        HIDDEN_CONTRACT_ID,
        SnapshotArchiveFeatureStore,
        SnapshotArchiveRefReader,
    )

    root = Path(args.output).resolve()
    root.mkdir(parents=True, exist_ok=False)
    report = {
        "status": "running",
        "source": "real serving captures replayed into live Mooncake TCP",
        "archives": args.archives,
        "target_transformer_prefill": False,
    }
    repository = Path(__file__).resolve().parents[2]
    refs = []
    for archive in args.archives:
        refs.extend(SnapshotArchiveRefReader(archive, run_id="hidden-smoke").read())
    if len({ref.sample_id for ref in refs}) != len(refs):
        raise ValueError("duplicate source samples")
    archive_reader = SnapshotArchiveFeatureStore()
    for ref in refs:
        if ref.metadata["contract_id"] != HIDDEN_CONTRACT_ID:
            raise ValueError("this smoke requires actual hidden-format samples")
        _, handle = archive_reader.get(ref)
        archive_reader.release(handle)
    archive_reader.close()
    report["validated_source_samples"] = len(refs)
    manifest = json.loads(Path(refs[0].metadata["manifest_path"]).read_bytes())
    draft = json.loads(
        (repository / "configs/qwen3.5-4b-dspark-target-kv.json").read_bytes()
    )
    draft["target_kv_contract"]["kv"] = manifest["kv"]
    draft["target_kv_contract"]["teacher"] = manifest["teacher"]
    draft["dflash_config"]["target_layer_ids"] = manifest["kv"]["selected_layer_ids"]
    draft_path = root / "draft-config.json"
    write_json(draft_path, draft)

    port = free_port()
    address = f"127.0.0.1:{port}"
    store = None
    with (root / "master.log").open("w") as log:
        master = subprocess.Popen(
            [
                "mooncake_master",
                f"--rpc_port={port}",
                f"--metrics_port={free_port()}",
                f"--http_metadata_server_port={free_port()}",
            ],
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        try:
            deadline = time.monotonic() + 30
            while True:
                if master.poll() is not None or time.monotonic() > deadline:
                    raise RuntimeError("Mooncake master did not start")
                try:
                    with socket.create_connection(("127.0.0.1", port), timeout=0.2):
                        break
                except OSError:
                    time.sleep(0.1)
            store = MooncakeDistributedStore()
            setup = {
                "local_hostname": f"127.0.0.1:{free_port()}",
                "metadata_server": "P2PHANDSHAKE",
                "master_server_addr": address,
                "protocol": "tcp",
                "rdma_devices": "",
                "global_segment_size": 256 << 20,
                "local_buffer_size": 32 << 20,
            }
            if store.setup(**setup) not in (None, 0):
                raise RuntimeError("Mooncake setup failed")
            publications = []
            for ref in refs:
                metadata = ref.metadata
                data = Path(metadata["manifest_path"]).read_bytes()
                item = json.loads(data)
                tensors = load_file(metadata["tensors_path"])
                for obj in item["objects"]:
                    payload = (
                        tensors[obj["key"]]
                        .contiguous()
                        .view(torch.uint8)
                        .numpy()
                        .tobytes()
                    )
                    if store.put(obj["key"], payload) not in (None, 0):
                        raise RuntimeError("Mooncake tensor write failed")
                key = f"draft-data/{item['dataset_id']}/{item['sample_id']}/{item['generation_id']}/manifest"
                if store.put(key, data) not in (None, 0):
                    raise RuntimeError("Mooncake manifest write failed")
                publications.append(
                    {
                        **{
                            field: item[field]
                            for field in (
                                "dataset_id",
                                "sample_id",
                                "generation_id",
                                "contract_id",
                            )
                        },
                        "manifest_key": key,
                        "manifest_sha256": hashlib.sha256(data).hexdigest(),
                        "manifest_nbytes": len(data),
                    }
                )
            refs_path = root / "publications.json"
            write_json(refs_path, publications)
            base = yaml.safe_load(
                (
                    repository
                    / "examples/configs/qwen3.5-4b-dspark-snapshot-archive.yaml"
                ).read_text()
            )
            base["model"]["target_model_path"] = args.model
            base["model"]["draft_model_config"] = str(draft_path)
            base["data"]["max_length"] = 64
            base["data"]["mooncake_snapshots"] = {
                "refs_path": str(refs_path),
                "metadata_server": "P2PHANDSHAKE",
                "master_server_addr": address,
                "protocol": "tcp",
                "local_hostname": "127.0.0.1",
                "local_buffer_size_bytes": 32 << 20,
            }
            base["run_id"] = "hidden-mooncake-smoke"
            base["output_dir"] = str(root / "checkpoints")
            base["training"].update(
                max_steps=2, total_steps=4, save_interval=2, max_checkpoints=2
            )
            evidence = []
            worker_env = os.environ.copy()
            for name in (
                "RANK",
                "WORLD_SIZE",
                "LOCAL_RANK",
                "MASTER_ADDR",
                "MASTER_PORT",
            ):
                worker_env.pop(name, None)
            for phase in ("train", "resume"):
                if phase == "resume":
                    base["training"].update(
                        max_steps=4,
                        resume_from=str(
                            root / "checkpoints/hidden-mooncake-smoke-step2"
                        ),
                    )
                config = root / f"{phase}.yaml"
                config.write_text(yaml.safe_dump(base))
                evidence_path = root / f"{phase}-evidence.json"
                with (root / f"{phase}.log").open("w") as train_log:
                    subprocess.run(
                        [
                            sys.executable,
                            str(Path(__file__).resolve()),
                            "--worker",
                            str(config),
                            "--evidence",
                            str(evidence_path),
                        ],
                        stdout=train_log,
                        stderr=subprocess.STDOUT,
                        check=True,
                        timeout=1200,
                        env=worker_env,
                    )
                evidence.append(json.loads(evidence_path.read_bytes()))
                print(f"{phase}: {evidence[-1]}", flush=True)
            assert [entry["step"] for run in evidence for entry in run["metrics"]] == [
                1,
                2,
                3,
                4,
            ]
            assert all(
                run["teacher_rows"] > 0
                and run["read_calls"] > 0
                and not run["teacher_has_grad"]
                for run in evidence
            )
            checkpoints = []
            for step in (2, 4):
                path = root / f"checkpoints/hidden-mooncake-smoke-step{step}"
                state = torch.load(
                    path / "training_state.pt",
                    map_location="cpu",
                    weights_only=False,
                    mmap=True,
                )
                assert state["global_step"] == step
                checkpoints.append(
                    {
                        "step": step,
                        "path": str(path),
                        "bytes": sum(
                            item.stat().st_size
                            for item in path.iterdir()
                            if item.is_file()
                        ),
                        "payload_format": state["snapshot_payload_format"],
                    }
                )
            report.update(
                status="passed",
                training=evidence,
                checkpoints=checkpoints,
                checkpoint_validation=verify_checkpoints(root),
                torch=torch.__version__,
                gpu=torch.cuda.get_device_name(),
            )
        except BaseException as error:
            report.update(status="failed", error=f"{type(error).__name__}: {error}")
            raise
        finally:
            write_json(root / "report.json", report)
            if store is not None:
                store.close()
            master.terminate()
            try:
                master.wait(timeout=10)
            except subprocess.TimeoutExpired:
                master.kill()
                master.wait()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--archives", nargs="+")
    parser.add_argument("--output")
    parser.add_argument(
        "--model", default="/gpfs/models/huggingface.co/Qwen/Qwen3.5-4B"
    )
    parser.add_argument("--worker")
    parser.add_argument("--evidence")
    args = parser.parse_args()
    if args.worker:
        worker(args.worker, args.evidence)
    else:
        run(args)
