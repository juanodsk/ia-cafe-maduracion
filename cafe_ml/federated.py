"""Three-client FedAvg simulation on frozen coffee-image embeddings.

Whole capture groups belong to one client. Client class distributions are
reported rather than assumed IID/non-IID. Sufficient statistics are aggregated
for a common train-only normalizer; this local simulation does not implement
secure aggregation, differential privacy, or separate physical institutions.

FedAvg reference: https://proceedings.mlr.press/v54/mcmahan17a.html
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import time
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import nn

from .data import CLASS_NAMES, load_records
from .evaluation import evaluate_predictions, save_evaluation
from .head_training import (
    _csv, _json, _study_signature, _validate_features, _validate_records,
    build_head, labels_from_records, predict_head, save_head_checkpoint,
    seed_everything,
)


def client_group_assignments(records: Sequence[Mapping[str, Any]], n_clients: int = 3,
                             seed: int = 42) -> np.ndarray:
    """Shuffle groups deterministically and round-robin complete groups."""
    groups = sorted({str(record["group_id"]) for record in records})
    if not 2 <= n_clients <= len(groups):
        raise ValueError("Federated learning requires at least two clients and enough capture groups.")
    order = np.random.default_rng(seed).permutation(len(groups))
    mapping = {groups[int(group_index)]: position % n_clients for position, group_index in enumerate(order)}
    return np.asarray([mapping[str(record["group_id"])] for record in records], dtype=np.int64)


def aggregate_training_statistics(features: np.ndarray, targets: np.ndarray,
                                   assignments: np.ndarray, n_clients: int
                                   ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Aggregate only per-client counts, sums and squared sums, in float64."""
    feature_sum = np.zeros(features.shape[1], dtype=np.float64)
    feature_square_sum = np.zeros(features.shape[1], dtype=np.float64)
    class_counts = np.zeros(len(CLASS_NAMES), dtype=np.int64)
    total = 0
    for client in range(n_clients):
        indices = np.flatnonzero(assignments == client)
        if indices.size == 0:
            raise ValueError("A simulated client has no training data.")
        client_sum = np.zeros(features.shape[1], dtype=np.float64)
        client_square_sum = np.zeros(features.shape[1], dtype=np.float64)
        for start in range(0, len(indices), 1024):
            batch = np.asarray(features[indices[start:start + 1024]], dtype=np.float64)
            client_sum += batch.sum(axis=0)
            client_square_sum += np.square(batch).sum(axis=0)
        feature_sum += client_sum
        feature_square_sum += client_square_sum
        class_counts += np.bincount(targets[indices], minlength=len(CLASS_NAMES))
        total += len(indices)
    mean = feature_sum / total
    variance = np.maximum(feature_square_sum / total - np.square(mean), 0)
    return mean.astype(np.float32), np.maximum(np.sqrt(variance), 1e-5).astype(np.float32), class_counts


def average_client_states(states: Sequence[Mapping[str, torch.Tensor]], sample_counts: Sequence[int],
                           reference: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """FedAvg weighted by client sample count; common normalization stays fixed."""
    if len(states) != len(sample_counts) or not states or any(n <= 0 for n in sample_counts):
        raise ValueError("FedAvg needs one positive sample count per client state.")
    total = sum(sample_counts)
    averaged = {}
    for name, value in reference.items():
        if name.startswith("feature_"):
            if any(not torch.equal(state[name], value) for state in states):
                raise ValueError("Clients changed the shared train-only normalization buffers.")
            averaged[name] = value.clone()
        else:
            result = torch.zeros_like(value, dtype=torch.float64)
            for state, count in zip(states, sample_counts):
                result += state[name].to(dtype=torch.float64) * (count / total)
            averaged[name] = result.to(dtype=value.dtype)
    return averaged


def _plot_rounds(path: Path, rows: list[dict[str, Any]]) -> None:
    import matplotlib
    matplotlib.use("Agg", force=True)
    from matplotlib import pyplot as plt
    figure, axis = plt.subplots(figsize=(7, 4.3), layout="constrained")
    axis.plot([r["round"] for r in rows], [r["valid_macro_f1"] for r in rows], marker="o", label="Macro F1")
    axis.plot([r["round"] for r in rows], [r["valid_accuracy"] for r in rows], marker="s", label="Accuracy")
    axis.set(xlabel="Ronda FedAvg", ylabel="Validación", title="FedAvg con tres clientes simulados", ylim=(0, 1))
    axis.grid(alpha=0.2)
    axis.legend()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def run_federated_experiment(train_features: np.ndarray, train_records: Sequence[Mapping[str, Any]],
                              valid_features: np.ndarray, valid_records: Sequence[Mapping[str, Any]],
                              output_dir: str | Path, *, params: Mapping[str, Any] | None = None,
                              n_clients: int = 3, rounds: int = 5, local_epochs: int = 2,
                              seed: int = 42, device: str = "cpu") -> dict[str, Any]:
    _validate_records(train_records, "train")
    _validate_records(valid_records, "valid")
    if set(r["group_id"] for r in train_records) & set(r["group_id"] for r in valid_records):
        raise ValueError("Federated training and validation groups overlap.")
    train_features = _validate_features(train_features, len(train_records))
    valid_features = _validate_features(valid_features, len(valid_records))
    if train_features.shape[1] != valid_features.shape[1] or rounds < 1 or local_epochs < 1:
        raise ValueError("Invalid feature dimensions or round/epoch budgets.")
    params = {"lr": 1e-3, "batch_size": 256, "weight_decay": 1e-4, "dropout": 0.2,
              "hidden_dim": 0, "class_weight_power": 0.5, **dict(params or {})}
    config = {"method": "FedAvg", "simulation": True, "n_clients": n_clients, "rounds": rounds,
              "local_epochs": local_epochs, "seed": seed, "device": device, "params": params,
              "data_signature": _study_signature(train_features, train_records, valid_features, valid_records)}
    output_dir = Path(output_dir).resolve()
    summary_path = output_dir / "summary.json"
    if summary_path.exists():
        previous = json.loads(summary_path.read_text(encoding="utf-8"))
        if previous["config"] != config:
            raise ValueError("Existing federated result has different data/settings; choose a new output directory.")
        if not Path(previous["checkpoint_path"]).exists():
            raise ValueError("Federated summary exists but its saved checkpoint is missing.")
        return previous
    output_dir.mkdir(parents=True, exist_ok=True)
    _json(output_dir / "config.json", config)
    seed_everything(seed)
    torch.set_num_threads(4)
    targets, valid_targets = labels_from_records(train_records), labels_from_records(valid_records)
    assignments = client_group_assignments(train_records, n_clients, seed)
    mean, scale, class_counts = aggregate_training_statistics(train_features, targets, assignments, n_clients)
    if np.any(class_counts == 0):
        raise ValueError("All maturity classes must be present across the training clients.")
    power = float(params["class_weight_power"])
    weights = (len(targets) / (len(CLASS_NAMES) * class_counts)) ** power
    weights = (weights / weights.mean()).astype(np.float32)
    distribution = []
    for client in range(n_clients):
        indices = np.flatnonzero(assignments == client)
        counts = np.bincount(targets[indices], minlength=len(CLASS_NAMES))
        distribution.append({"client_id": client, "samples": len(indices),
                             "groups": len({train_records[int(i)]["group_id"] for i in indices}),
                             **{label: int(counts[c]) for c, label in enumerate(CLASS_NAMES)}})
    _csv(output_dir / "client_distribution.csv", distribution)
    _csv(output_dir / "client_assignments.csv", [{"row_index": i, "client_id": int(assignments[i]),
          "split": "train", "group_id": record["group_id"], "image_id": record.get("image_id"),
          "annotation_id": record.get("annotation_id"), "crop_id": record.get("crop_id"), "label": record.get("label")}
          for i, record in enumerate(train_records)])
    global_head = build_head(train_features.shape[1], params)
    global_head.feature_mean.copy_(torch.from_numpy(mean))
    global_head.feature_scale.copy_(torch.from_numpy(scale))
    global_head.to(device)
    rows, client_loss_rows = [], []
    best_score, best_round = -1.0, 0
    started = time.monotonic()
    for round_index in range(0, rounds + 1):
        if round_index:
            states, counts, losses = [], [], []
            reference = {key: value.detach().cpu().clone() for key, value in global_head.state_dict().items()}
            for client in range(n_clients):
                indices = np.flatnonzero(assignments == client)
                local = copy.deepcopy(global_head).to(device).train()
                seed_everything(seed + 1000 * round_index + client)
                x = torch.tensor(np.asarray(train_features[indices], dtype=np.float32), device=device)
                y = torch.tensor(targets[indices], dtype=torch.long, device=device)
                criterion = nn.CrossEntropyLoss(weight=torch.tensor(weights, device=device))
                optimizer = torch.optim.SGD(local.parameters(), lr=float(params["lr"]), weight_decay=float(params["weight_decay"]))
                generator = torch.Generator().manual_seed(seed + 1000 * round_index + client)
                loss_sum, examples = 0.0, 0
                for _ in range(local_epochs):
                    order = torch.randperm(len(indices), generator=generator).to(device)
                    for start in range(0, len(indices), int(params["batch_size"])):
                        batch = order[start:start + int(params["batch_size"])]
                        optimizer.zero_grad(set_to_none=True)
                        loss = criterion(local(x[batch]), y[batch])
                        if not torch.isfinite(loss):
                            raise FloatingPointError("Nonfinite federated local loss.")
                        loss.backward()
                        optimizer.step()
                        loss_sum += float(loss.detach()) * len(batch)
                        examples += len(batch)
                states.append({key: value.detach().cpu().clone() for key, value in local.state_dict().items()})
                counts.append(len(indices))
                losses.append(loss_sum / examples)
                client_loss_rows.append({"round": round_index, "client_id": client, "local_epochs": local_epochs,
                                         "training_samples": len(indices), "train_loss": loss_sum / examples})
                del local, x, y, optimizer
            averaged = average_client_states(states, counts, reference)
            global_head.load_state_dict(averaged, strict=True)
            training_loss = float(np.average(losses, weights=counts))
        else:
            training_loss = None
        probabilities = predict_head(global_head, valid_features, device=device)
        metrics = evaluate_predictions(valid_targets, probabilities)
        rows.append({"round": round_index, "train_loss": training_loss, "valid_macro_f1": metrics["macro_f1"],
                     "valid_accuracy": metrics["accuracy"], "valid_balanced_accuracy": metrics["balanced_accuracy"],
                     "elapsed_seconds": time.monotonic() - started})
        if round_index and metrics["macro_f1"] > best_score:
            best_score, best_round = metrics["macro_f1"], round_index
            save_head_checkpoint(output_dir / "checkpoint.pt", global_head,
                                 {"method": "FedAvg", "simulation": True, "round": round_index,
                                  "normalization": "aggregated_training_client_sums_and_squared_sums", "config": config})
        _csv(output_dir / "round_metrics.csv", rows)
        _csv(output_dir / "client_training_losses.csv", client_loss_rows,
             ["round", "client_id", "local_epochs", "training_samples", "train_loss"])
        print(f"[FedAvg] round {round_index}/{rounds}: validation macro-F1={metrics['macro_f1']:.5f}", flush=True)
    save_head_checkpoint(output_dir / "final_round_checkpoint.pt", global_head, {"method": "FedAvg", "round": rounds, "config": config})
    from .head_training import load_head_checkpoint
    selected = load_head_checkpoint(output_dir / "checkpoint.pt", device)
    validation_metrics = save_evaluation(output_dir / "validation", valid_records, predict_head(selected, valid_features, device=device),
                                        model_name="FedAvg_simulated_clients", metadata={"selected_round": best_round, "test_used": False})
    _plot_rounds(output_dir / "round_curves.png", rows)
    summary = {"method": "FedAvg", "config": config, "checkpoint_path": str(output_dir / "checkpoint.pt"),
               "best_round": best_round, "validation_metrics": validation_metrics, "client_distribution": distribution,
               "class_weights": weights.tolist(), "test_accessed": False, "elapsed_seconds": time.monotonic() - started,
               "limitations": ["Three clients are simulated on one computer, not three real farms.",
                               "Whole capture groups are assigned after a seeded shuffle; actual class distributions are reported without assuming strong non-IID skew.",
                               "Aggregate statistics are ordinary local sums; secure aggregation and differential privacy are not implemented.",
                               "Only classifier heads are trained; the ImageNet backbone remains frozen."]}
    _json(summary_path, summary)
    _json(output_dir / "metrics.json", validation_metrics)
    return summary


def load_cached_partitions(features_dir: str | Path, dataset_root: str | Path):
    """Load train/valid only, checking each cache's saved row identity digest."""
    from .experiment import record_digest, digest_file
    features_dir = Path(features_dir)
    records = [load_records(dataset_root, split) for split in ("train", "valid")]
    arrays = []
    for split, rows in zip(("train", "valid"), records):
        feature_path, metadata_path = features_dir / f"{split}.npy", features_dir / f"{split}.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata["configuration"]["records_sha256"] != record_digest(rows):
            raise ValueError(f"Cached {split} feature rows do not match the saved dataset records.")
        if metadata["feature_file_sha256"] != digest_file(feature_path):
            raise ValueError(f"Cached {split} features have changed.")
        arrays.append(np.load(feature_path, mmap_mode="r"))
    return arrays[0], records[0], arrays[1], records[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, default=Path("dataset_preparado"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--clients", type=int, default=3)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--local-epochs", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    values = load_cached_partitions(args.features, args.dataset)
    result = run_federated_experiment(*values, args.output, n_clients=args.clients, rounds=args.rounds,
                                      local_epochs=args.local_epochs, seed=args.seed)
    print(json.dumps({"method": result["method"], "best_round": result["best_round"],
                      "valid_macro_f1": result["validation_metrics"]["macro_f1"]}, indent=2))


if __name__ == "__main__":
    main()
