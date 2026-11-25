"""Real CPU DistributedDataParallel pilot using two local Gloo processes.

This is one-machine data-parallel training of a classifier on cached frozen
features, not a multi-GPU or multi-machine benchmark. The fixed epoch budget
and effective global batch size differ from the Hyperband baseline study.

Reference: https://docs.pytorch.org/tutorials/intermediate/ddp_tutorial.html
"""

from __future__ import annotations

import argparse
from datetime import timedelta
import hashlib
import json
import math
import os
from pathlib import Path
import time
from typing import Any, Mapping, Sequence
import uuid

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler, TensorDataset

from .data import CLASS_NAMES
from .evaluation import evaluate_predictions, save_evaluation
from .federated import load_cached_partitions
from .head_training import (
    _csv, _json, _study_signature, _validate_features, _validate_records,
    build_head, labels_from_records, predict_head, save_head_checkpoint,
    seed_everything,
)


def _feature_file(array: np.ndarray, fallback: Path) -> Path:
    filename = getattr(array, "filename", None)
    if filename is not None:
        existing = Path(filename)
        if existing.exists():
            stored = np.load(existing, mmap_mode="r")
            if stored.shape == array.shape and stored.dtype == array.dtype and np.array_equal(stored, array):
                return existing.resolve()
    np.save(fallback, array, allow_pickle=False)
    return fallback.resolve()


def _state_digest(model: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _plot_epochs(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    import matplotlib
    matplotlib.use("Agg", force=True)
    from matplotlib import pyplot as plt
    figure, axis = plt.subplots(figsize=(6.8, 4.2), layout="constrained")
    axis.plot([r["epoch"] for r in rows], [r["valid_macro_f1"] for r in rows], marker="o", label="Macro F1")
    axis.plot([r["epoch"] for r in rows], [r["valid_accuracy"] for r in rows], marker="s", label="Accuracy")
    axis.set(xlabel="Época", ylabel="Validación", title="DDP con dos procesos CPU y Gloo", ylim=(0, 1))
    axis.grid(alpha=0.2)
    axis.legend()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def _ddp_worker(rank: int, world_size: int, launch_path: str, rendezvous_uri: str) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ.setdefault("MPLCONFIGDIR", str(Path(launch_path).parent / "matplotlib_cache"))
    config = json.loads(Path(launch_path).read_text(encoding="utf-8"))
    output_dir = Path(config["output_dir"])
    torch.set_num_threads(2)
    seed_everything(config["seed"])
    dist.init_process_group("gloo", init_method=rendezvous_uri, rank=rank,
                            world_size=world_size, timeout=timedelta(seconds=90))
    try:
        features = np.load(config["train_features"], mmap_mode="r")
        targets = np.load(config["train_targets"], allow_pickle=False)
        x = torch.tensor(np.asarray(features, dtype=np.float32), dtype=torch.float32)
        y = torch.tensor(targets, dtype=torch.long)
        unique_indices = torch.arange(rank, len(y), world_size)
        local = x[unique_indices].double()
        feature_sum = local.sum(dim=0)
        square_sum = local.square().sum(dim=0)
        count = torch.tensor([len(unique_indices)], dtype=torch.float64)
        class_counts = torch.bincount(y[unique_indices], minlength=len(CLASS_NAMES)).to(dtype=torch.float64)
        for statistics in (feature_sum, square_sum, count, class_counts):
            dist.all_reduce(statistics, op=dist.ReduceOp.SUM)
        mean = feature_sum / count
        scale = (square_sum / count - mean.square()).clamp_min(0).sqrt().clamp_min(1e-5)
        if torch.any(class_counts == 0):
            raise ValueError("Every class must occur in distributed training data.")
        weights = (len(y) / (len(CLASS_NAMES) * class_counts)) ** float(config["params"]["class_weight_power"])
        weights = (weights / weights.mean()).float()
        head = build_head(x.shape[1], config["params"])
        head.feature_mean.copy_(mean.float())
        head.feature_scale.copy_(scale.float())
        del local
        ddp = DistributedDataParallel(head, broadcast_buffers=False)
        # Distinct dropout streams after DDP broadcasts identical initial state.
        torch.manual_seed(config["seed"] + rank + 1)
        dataset = TensorDataset(x, y)
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank,
                                     shuffle=True, seed=config["seed"], drop_last=False)
        loader = DataLoader(dataset, batch_size=config["batch_size"], sampler=sampler, num_workers=0)
        criterion = nn.CrossEntropyLoss(weight=weights)
        optimizer = torch.optim.AdamW(ddp.parameters(), lr=float(config["params"]["lr"]),
                                      weight_decay=float(config["params"]["weight_decay"]))
        rows = []
        first_indices = None
        steps = 0
        started = time.monotonic()
        for epoch in range(1, config["epochs"] + 1):
            sampler.set_epoch(epoch)
            if first_indices is None:
                first_indices = list(iter(sampler))
            ddp.train()
            local_loss, local_examples = 0.0, 0
            for batch, target in loader:
                optimizer.zero_grad(set_to_none=True)
                loss = criterion(ddp(batch), target)
                if not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite distributed training loss.")
                loss.backward()  # Real Gloo all-reduce of parameter gradients.
                optimizer.step()
                local_loss += float(loss.detach()) * len(batch)
                local_examples += len(batch)
                steps += 1
            training_totals = torch.tensor([local_loss, local_examples], dtype=torch.float64)
            dist.all_reduce(training_totals, op=dist.ReduceOp.SUM)
            if rank == 0:
                validation = np.load(config["valid_features"], mmap_mode="r")
                valid_targets = np.load(config["valid_targets"], allow_pickle=False)
                probabilities = predict_head(ddp.module, validation, device="cpu")
                metrics = evaluate_predictions(valid_targets, probabilities)
                rows.append({"epoch": epoch, "train_loss": float(training_totals[0] / training_totals[1]),
                             "processed_training_rows": int(training_totals[1]), "valid_macro_f1": metrics["macro_f1"],
                             "valid_accuracy": metrics["accuracy"], "valid_balanced_accuracy": metrics["balanced_accuracy"],
                             "elapsed_seconds": time.monotonic() - started})
                _csv(output_dir / "epoch_metrics.csv", rows)
                print(f"[DDP CPU/Gloo] epoch {epoch}/{config['epochs']}: validation macro-F1={metrics['macro_f1']:.5f}", flush=True)
            dist.barrier()
        digest = _state_digest(ddp.module)
        all_digests = [None] * world_size
        dist.all_gather_object(all_digests, digest)
        if len(set(all_digests)) != 1:
            raise AssertionError("DDP ranks ended with different model states.")
        _json(output_dir / f"rank_{rank}.json", {"rank": rank, "pid": os.getpid(), "world_size": world_size,
              "backend": "gloo", "device": "cpu", "optimizer_steps": steps, "epochs": config["epochs"],
              "samples_per_epoch": len(sampler), "first_epoch_indices": first_indices, "state_sha256": digest})
        if rank == 0:
            records = json.loads(Path(config["valid_records"]).read_text(encoding="utf-8"))
            checkpoint_path = output_dir / "checkpoint.pt"
            save_head_checkpoint(checkpoint_path, ddp.module, {"method": "DistributedDataParallel", "backend": "gloo",
                                 "world_size": world_size, "device": "cpu", "epochs": config["epochs"],
                                 "normalization": "all_reduced_unique_training_shard_sums", "config": config["public_config"]})
            probabilities = predict_head(ddp.module, np.load(config["valid_features"], mmap_mode="r"), device="cpu")
            validation_metrics = save_evaluation(output_dir / "validation", records, probabilities,
                                                 model_name="DDP_CPU_Gloo", metadata={"world_size": world_size, "epochs": config["epochs"], "test_used": False})
            _plot_epochs(output_dir / "epoch_curves.png", rows)
            result = {"method": "DistributedDataParallel", "config": config["public_config"],
                      "checkpoint_path": str(checkpoint_path), "validation_metrics": validation_metrics,
                      "rank_model_state_sha256": all_digests, "rank_states_identical": True,
                      "effective_global_batch_size": config["batch_size"] * world_size,
                      "sampler_padding_rows_per_epoch": math.ceil(len(y) / world_size) * world_size - len(y),
                      "test_accessed": False, "elapsed_seconds": time.monotonic() - started,
                      "limitations": ["Two real CPU processes run on the same machine; this is not a multi-GPU or multi-machine speed benchmark.",
                                      "Only the classifier head is updated; the pretrained backbone is frozen.",
                                      "The fixed two-epoch pilot budget is descriptive and differs from tuned baseline training."]}
            _json(output_dir / "summary.json", result)
            _json(output_dir / "metrics.json", validation_metrics)
        dist.barrier()
    finally:
        dist.destroy_process_group()


def run_distributed_experiment(train_features: np.ndarray, train_records: Sequence[Mapping[str, Any]],
                                valid_features: np.ndarray, valid_records: Sequence[Mapping[str, Any]],
                                output_dir: str | Path, *, params: Mapping[str, Any] | None = None,
                                world_size: int = 2, epochs: int = 2, seed: int = 42) -> dict[str, Any]:
    if not dist.is_available() or not dist.is_gloo_available():
        raise RuntimeError("This PyTorch installation has no Gloo distributed backend.")
    if world_size < 2 or epochs < 1:
        raise ValueError("DDP requires at least two processes and a positive epoch budget.")
    _validate_records(train_records, "train")
    _validate_records(valid_records, "valid")
    if set(r["group_id"] for r in train_records) & set(r["group_id"] for r in valid_records):
        raise ValueError("Distributed training and validation groups overlap.")
    train_features = _validate_features(train_features, len(train_records))
    valid_features = _validate_features(valid_features, len(valid_records))
    if train_features.shape[1] != valid_features.shape[1] or len(train_records) < world_size:
        raise ValueError("Invalid distributed feature dimensions or shard size.")
    params = {"lr": 1e-3, "batch_size": 256, "weight_decay": 1e-4, "dropout": 0.2,
              "hidden_dim": 0, "class_weight_power": 0.5, **dict(params or {})}
    public_config = {"method": "DistributedDataParallel", "backend": "gloo", "device": "cpu",
                     "torch_version": str(torch.__version__),
                     "world_size": world_size, "epochs": epochs, "seed": seed, "params": params,
                     "data_signature": _study_signature(train_features, train_records, valid_features, valid_records)}
    output_dir = Path(output_dir).resolve()
    summary_path = output_dir / "summary.json"
    if summary_path.exists():
        previous = json.loads(summary_path.read_text(encoding="utf-8"))
        if previous["config"] != public_config:
            raise ValueError("Existing DDP result has different data/settings; choose a new output directory.")
        if not Path(previous["checkpoint_path"]).exists():
            raise ValueError("DDP summary exists but its saved checkpoint is missing.")
        return previous
    output_dir.mkdir(parents=True, exist_ok=True)
    _json(output_dir / "config.json", public_config)
    input_dir = output_dir / "inputs"
    input_dir.mkdir(exist_ok=True)
    train_path = _feature_file(train_features, input_dir / "train_features.npy")
    valid_path = _feature_file(valid_features, input_dir / "valid_features.npy")
    train_targets, valid_targets = labels_from_records(train_records), labels_from_records(valid_records)
    np.save(input_dir / "train_targets.npy", train_targets, allow_pickle=False)
    np.save(input_dir / "valid_targets.npy", valid_targets, allow_pickle=False)
    _json(input_dir / "valid_records.json", list(valid_records))
    launch = {"output_dir": str(output_dir), "seed": seed, "epochs": epochs, "params": params,
              "batch_size": int(params["batch_size"]), "train_features": str(train_path), "valid_features": str(valid_path),
              "train_targets": str(input_dir / "train_targets.npy"), "valid_targets": str(input_dir / "valid_targets.npy"),
              "valid_records": str(input_dir / "valid_records.json"), "public_config": public_config}
    launch_path = output_dir / "launch.json"
    _json(launch_path, launch)
    rendezvous = output_dir / f"gloo_store_{uuid.uuid4().hex}"
    # FileStore is explicitly supported by Windows Gloo and avoids depending on
    # the optional libuv TCPStore build. The store path must not already exist.
    mp.spawn(_ddp_worker, args=(world_size, str(launch_path), rendezvous.as_uri()), nprocs=world_size, join=True)
    ranks = [json.loads((output_dir / f"rank_{rank}.json").read_text(encoding="utf-8")) for rank in range(world_size)]
    if len({row["pid"] for row in ranks}) != world_size:
        raise AssertionError("DDP did not use distinct worker processes.")
    if len({row["state_sha256"] for row in ranks}) != 1:
        raise AssertionError("Saved DDP rank state hashes disagree.")
    distribution = []
    for rank_info in ranks:
        indices = rank_info["first_epoch_indices"]
        counts = np.bincount(train_targets[indices], minlength=len(CLASS_NAMES))
        distribution.append({"rank": rank_info["rank"], "pid": rank_info["pid"], "first_epoch_samples": len(indices),
                             "unique_rows": len(set(indices)), **{label: int(counts[c]) for c, label in enumerate(CLASS_NAMES)}})
    _csv(output_dir / "rank_distribution.csv", distribution)
    _csv(output_dir / "rank_assignments_first_epoch.csv", [{"rank": row["rank"], "position": position,
          "row_index": index, "split": "train", "group_id": train_records[index]["group_id"],
          "image_id": train_records[index].get("image_id"), "annotation_id": train_records[index].get("annotation_id")}
          for row in ranks for position, index in enumerate(row["first_epoch_indices"])])
    return json.loads(summary_path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, default=Path("dataset_preparado"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--world-size", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    values = load_cached_partitions(args.features, args.dataset)
    result = run_distributed_experiment(*values, args.output, world_size=args.world_size, epochs=args.epochs, seed=args.seed)
    print(json.dumps({"method": result["method"], "valid_macro_f1": result["validation_metrics"]["macro_f1"],
                      "rank_states_identical": result["rank_states_identical"]}, indent=2))


if __name__ == "__main__":
    main()
