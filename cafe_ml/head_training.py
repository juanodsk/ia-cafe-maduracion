"""Hyperband and grouped CV for learned heads on frozen ImageNet features.

The backbone is never updated here. Normalization and class weights are fitted
only on the training rows of each fit. The fixed validation split selects the
head and epoch; grouped CV uses train only, with that epoch budget held fixed.
No function in this module accepts records from the reserved test split.

References:
https://optuna.readthedocs.io/en/stable/reference/generated/optuna.pruners.HyperbandPruner.html
https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedGroupKFold.html
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path
import random
import time
from typing import Any, Mapping, Sequence

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import optuna
from sklearn.model_selection import StratifiedGroupKFold
import torch
from torch import nn

from .data import CLASS_NAMES, CLASS_TO_INDEX
from .evaluation import evaluate_predictions, save_evaluation


def _json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = list(rows[0]) if rows else []
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


def _device(value: str | torch.device) -> torch.device:
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; explicitly choose device='cpu' to run on CPU.")
    return device


def labels_from_records(records: Sequence[Mapping[str, Any]]) -> np.ndarray:
    labels = []
    for record in records:
        label = record.get("label")
        index = record.get("class_index", record.get("label_index"))
        if label is not None:
            if label not in CLASS_TO_INDEX:
                raise ValueError(f"Unknown maturity class: {label!r}")
            expected = CLASS_TO_INDEX[label]
            if index is not None and int(index) != expected:
                raise ValueError("Record class index differs from its label.")
            index = expected
        if index is None or not 0 <= int(index) < len(CLASS_NAMES):
            raise ValueError("Every record needs a valid label or class_index.")
        labels.append(int(index))
    return np.asarray(labels, dtype=np.int64)


def _validate_records(records: Sequence[Mapping[str, Any]], split: str) -> None:
    if not records or any(record.get("split") != split for record in records):
        raise ValueError(f"Expected only nonempty {split!r} records; test records are reserved.")
    if any(not record.get("group_id") for record in records):
        raise ValueError("Every record requires its saved parent group_id.")


def _validate_features(features: np.ndarray, expected_rows: int) -> np.ndarray:
    features = np.asarray(features)
    if features.ndim != 2 or features.shape[0] != expected_rows or features.shape[1] < 1:
        raise ValueError("Features must be an N x D matrix aligned with the record order.")
    if not np.issubdtype(features.dtype, np.floating) or not np.isfinite(features).all():
        raise ValueError("Embeddings must be finite floating point values.")
    return features


def compute_class_weights(targets: np.ndarray, power: float = 0.5) -> np.ndarray:
    targets = np.asarray(targets, dtype=np.int64)
    if targets.ndim != 1 or targets.size == 0 or np.any((targets < 0) | (targets >= len(CLASS_NAMES))):
        raise ValueError("Invalid training targets.")
    counts = np.bincount(targets, minlength=len(CLASS_NAMES)).astype(np.float64)
    if np.any(counts == 0):
        raise ValueError("Every maturity class must occur in the training subset.")
    if not 0 <= power <= 1:
        raise ValueError("class_weight_power must be between zero and one.")
    weights = (targets.size / (len(CLASS_NAMES) * counts)) ** power
    return (weights / weights.mean()).astype(np.float32)


class EmbeddingHead(nn.Module):
    """A saved train-only standardizer followed by dropout and a classifier."""

    def __init__(self, feature_dim: int, params: Mapping[str, Any], num_classes: int = 5):
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.params = dict(params)
        self.num_classes = int(num_classes)
        self.register_buffer("feature_mean", torch.zeros(self.feature_dim))
        self.register_buffer("feature_scale", torch.ones(self.feature_dim))
        hidden = int(params.get("hidden_dim", 0))
        dropout = float(params.get("dropout", 0.2))
        if self.feature_dim < 1 or self.num_classes != len(CLASS_NAMES) or hidden < 0 or not 0 <= dropout < 1:
            raise ValueError("Invalid feature dimensions, class count or head hyperparameters.")
        if hidden:
            self.layers = nn.Sequential(nn.Linear(self.feature_dim, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, num_classes))
        else:
            self.layers = nn.Sequential(nn.Dropout(dropout), nn.Linear(self.feature_dim, num_classes))
        self.checkpoint_metadata: dict[str, Any] = {}

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.layers((features - self.feature_mean) / self.feature_scale)

    @torch.no_grad()
    def fit_normalization(self, training_features: torch.Tensor) -> None:
        variance, mean = torch.var_mean(training_features, dim=0, unbiased=False)
        self.feature_mean.copy_(mean)
        self.feature_scale.copy_(variance.sqrt().clamp_min(1e-5))


def build_head(feature_dim: int, params: Mapping[str, Any], num_classes: int = 5) -> EmbeddingHead:
    return EmbeddingHead(feature_dim, params, num_classes)


def _cpu_state(head: nn.Module) -> dict[str, torch.Tensor]:
    return {name: tensor.detach().cpu().clone() for name, tensor in head.state_dict().items()}


def save_head_checkpoint(path: str | Path, head: EmbeddingHead, metadata: Mapping[str, Any] | None = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {"schema_version": 1, "feature_dim": head.feature_dim, "num_classes": head.num_classes,
                  "params": head.params, "class_labels": list(CLASS_NAMES), "state_dict": _cpu_state(head),
                  "metadata": dict(metadata or {})}
    torch.save(checkpoint, path)


def load_head_checkpoint(path: str | Path, device: str | torch.device = "cpu") -> EmbeddingHead:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint["class_labels"] != list(CLASS_NAMES):
        raise ValueError("Checkpoint class vocabulary/order differs from the project.")
    head = build_head(checkpoint["feature_dim"], checkpoint["params"], checkpoint["num_classes"])
    head.load_state_dict(checkpoint["state_dict"], strict=True)
    head.checkpoint_metadata = dict(checkpoint.get("metadata", {}))
    return head.to(_device(device)).eval()


@torch.inference_mode()
def predict_head(head: EmbeddingHead, features: np.ndarray | torch.Tensor,
                 batch_size: int = 256, device: str | torch.device = "cpu") -> np.ndarray:
    if batch_size < 1 or len(features) == 0:
        raise ValueError("Prediction requires a nonempty feature matrix and positive batch size.")
    if features.ndim != 2 or features.shape[1] != head.feature_dim:
        raise ValueError("Prediction feature dimension differs from the trained head.")
    device = _device(device)
    head.to(device).eval()
    probabilities = []
    for start in range(0, len(features), batch_size):
        batch = features[start:start + batch_size]
        if not isinstance(batch, torch.Tensor):
            batch = torch.from_numpy(np.array(batch, dtype=np.float32, copy=True))
        batch = batch.to(device=device, dtype=torch.float32)
        probabilities.append(head(batch).softmax(dim=1).cpu().numpy())
    return np.concatenate(probabilities).astype(np.float32, copy=False)


def train_head(train_features: np.ndarray, train_targets: np.ndarray,
               valid_features: np.ndarray, valid_targets: np.ndarray,
               params: Mapping[str, Any], *, device: str | torch.device = "cpu",
               seed: int = 42, max_epochs: int = 12, trial: optuna.Trial | None = None,
               select_best: bool = True, output_dir: str | Path | None = None
               ) -> tuple[EmbeddingHead, list[dict[str, Any]], dict[str, Any]]:
    """Fit a head; CV sets select_best=False to evaluate a fixed final epoch."""
    if max_epochs < 1:
        raise ValueError("max_epochs must be positive.")
    train_targets = np.asarray(train_targets, dtype=np.int64)
    valid_targets = np.asarray(valid_targets, dtype=np.int64)
    train_features = _validate_features(train_features, len(train_targets))
    valid_features = _validate_features(valid_features, len(valid_targets))
    if len(valid_targets) == 0 or train_features.shape[1] != valid_features.shape[1]:
        raise ValueError("Train and validation feature dimensions must match and have rows.")
    device = _device(device)
    seed_everything(seed)
    head = build_head(train_features.shape[1], params).to(device)
    x_train = torch.tensor(np.asarray(train_features, dtype=np.float32), dtype=torch.float32, device=device)
    x_valid = torch.tensor(np.asarray(valid_features, dtype=np.float32), dtype=torch.float32, device=device)
    y_train = torch.tensor(train_targets, dtype=torch.long, device=device)
    head.fit_normalization(x_train)
    weights = compute_class_weights(train_targets, float(params.get("class_weight_power", 0.5)))
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(weights, device=device))
    optimizer = torch.optim.AdamW(head.parameters(), lr=float(params.get("lr", 1e-3)), weight_decay=float(params.get("weight_decay", 1e-4)))
    batch_size = int(params.get("batch_size", 256))
    if batch_size < 1:
        raise ValueError("batch_size must be positive.")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    history = []
    best_metrics: dict[str, Any] = {"macro_f1": -1.0}
    best_state = None
    started = time.perf_counter()
    for epoch in range(1, max_epochs + 1):
        head.train()
        order = torch.randperm(len(train_targets), generator=generator).to(device)
        train_loss = 0.0
        for start in range(0, len(order), batch_size):
            indices = order[start:start + batch_size]
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(head(x_train[indices]), y_train[indices])
            if not torch.isfinite(loss):
                raise FloatingPointError("Head training produced a nonfinite loss.")
            loss.backward()
            optimizer.step()
            train_loss += float(loss.detach()) * len(indices)
        probabilities = predict_head(head, x_valid, batch_size=batch_size, device=device)
        metrics = evaluate_predictions(valid_targets, probabilities)
        row = {"epoch": epoch, "train_loss": train_loss / len(order), "valid_macro_f1": metrics["macro_f1"],
               "valid_accuracy": metrics["accuracy"], "valid_balanced_accuracy": metrics["balanced_accuracy"],
               "elapsed_seconds": time.perf_counter() - started}
        history.append(row)
        if not select_best or metrics["macro_f1"] > best_metrics["macro_f1"]:
            best_metrics = {**metrics, "epoch": epoch, "class_weights": weights.tolist(), "seed": seed}
            best_state = _cpu_state(head)
        if output_dir is not None:
            _csv(Path(output_dir) / "history.csv", history)
        if trial is not None:
            trial.set_user_attr("best_epoch", int(best_metrics["epoch"]))
            trial.set_user_attr("training_seed", int(seed))
            trial.report(metrics["macro_f1"], step=epoch)
            if trial.should_prune():
                raise optuna.TrialPruned(f"Pruned at epoch {epoch}; macro_f1={metrics['macro_f1']:.5f}")
    head.load_state_dict(best_state, strict=True)
    head.eval()
    return head, history, best_metrics


def _study_signature(train_features: np.ndarray, train_records: Sequence[Mapping[str, Any]],
                     valid_features: np.ndarray, valid_records: Sequence[Mapping[str, Any]]) -> str:
    digest = hashlib.sha256()
    for features, records in ((train_features, train_records), (valid_features, valid_records)):
        digest.update(str(features.shape).encode())
        digest.update(str(features.dtype).encode())
        digest.update(np.ascontiguousarray(features).view(np.uint8))
        identity = [(r.get("crop_id"), r.get("image_id"), r.get("annotation_id"), r["group_id"], r.get("label"), r["split"]) for r in records]
        digest.update(json.dumps(identity, separators=(",", ":")).encode())
    return digest.hexdigest()


def tune_head(train_features: np.ndarray, train_records: Sequence[Mapping[str, Any]],
              valid_features: np.ndarray, valid_records: Sequence[Mapping[str, Any]],
              output_dir: str | Path, *, device: str = "cpu", seed: int = 42,
              n_trials: int = 12, max_epochs: int = 12, model_name: str | None = None) -> dict[str, Any]:
    if n_trials < 1 or max_epochs < 1:
        raise ValueError("Tuning requires positive trial and epoch budgets.")
    _validate_records(train_records, "train")
    _validate_records(valid_records, "valid")
    if set(r["group_id"] for r in train_records) & set(r["group_id"] for r in valid_records):
        raise ValueError("Training and holdout validation groups overlap.")
    train_features = _validate_features(train_features, len(train_records))
    valid_features = _validate_features(valid_features, len(valid_records))
    train_targets, valid_targets = labels_from_records(train_records), labels_from_records(valid_records)
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    study_name = f"coffee-head-{model_name or output_dir.name}-seed-{seed}"
    # Optuna formats storage URL templates using str.format(SCHEMA_VERSION=...).
    # Escape literal braces in Windows/sandbox paths before that formatting.
    database_path = (output_dir / "study.sqlite3").as_posix().replace("{", "{{").replace("}", "}}")
    storage = "sqlite:///" + database_path
    backend = optuna.storages.RDBStorage(url=storage)
    try:
        study = optuna.create_study(study_name=study_name, direction="maximize", storage=backend, load_if_exists=True,
                                   sampler=optuna.samplers.RandomSampler(seed=seed),
                                   pruner=optuna.pruners.HyperbandPruner(min_resource=min(2, max_epochs), max_resource=max_epochs, reduction_factor=3))
        return _tune_with_study(study, train_features, train_records, valid_features, valid_records, output_dir,
                                device=device, seed=seed, n_trials=n_trials, max_epochs=max_epochs, model_name=model_name)
    finally:
        # Explicitly release SQLite handles, including on failed/interrupted fits.
        # Windows otherwise keeps the database locked through SQLAlchemy's pool.
        backend.remove_session()
        backend.engine.dispose()


def _tune_with_study(study: optuna.Study, train_features: np.ndarray, train_records: Sequence[Mapping[str, Any]],
                     valid_features: np.ndarray, valid_records: Sequence[Mapping[str, Any]], output_dir: Path,
                     *, device: str, seed: int, n_trials: int, max_epochs: int, model_name: str | None) -> dict[str, Any]:
    train_targets, valid_targets = labels_from_records(train_records), labels_from_records(valid_records)
    study_name = study.study_name
    signature = _study_signature(train_features, train_records, valid_features, valid_records)
    if study.user_attrs.get("data_signature", signature) != signature:
        raise ValueError("An existing study uses different feature data or record ordering; select a new output directory.")
    settings = {"seed": seed, "max_epochs": max_epochs, "sampler": "RandomSampler", "pruner": "HyperbandPruner", "min_resource": min(2, max_epochs), "reduction_factor": 3}
    if study.user_attrs.get("settings", settings) != settings:
        raise ValueError("An existing study uses different search settings; select a new output directory.")
    study.set_user_attr("data_signature", signature)
    study.set_user_attr("settings", settings)

    # RandomSampler's RNG lives in memory, not in SQLite. Replay its public
    # independent sampling calls to advance it past previously stored trials.
    # The parameter declaration order below is fixed by this search space.
    parameter_order = ("lr", "batch_size", "weight_decay", "dropout", "hidden_dim", "class_weight_power")
    for previous in sorted(study.trials, key=lambda trial: trial.number):
        for name in parameter_order:
            if name in previous.distributions:
                study.sampler.sample_independent(study, previous, name, previous.distributions[name])

    def objective(trial: optuna.Trial) -> float:
        params = {"lr": trial.suggest_float("lr", 1e-4, 3e-2, log=True),
                  "batch_size": trial.suggest_categorical("batch_size", [128, 256]),
                  "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True),
                  "dropout": trial.suggest_categorical("dropout", [0.0, 0.2]),
                  "hidden_dim": trial.suggest_categorical("hidden_dim", [0, 256]),
                  "class_weight_power": trial.suggest_categorical("class_weight_power", [0.5, 1.0])}
        trial_dir = output_dir / "trials" / f"trial_{trial.number:03d}"
        _, _, metrics = train_head(train_features, train_targets, valid_features, valid_targets, params,
                                   device=device, seed=seed + trial.number, max_epochs=max_epochs,
                                   trial=trial, output_dir=trial_dir)
        _json(trial_dir / "metrics.json", metrics)
        print(f"[{model_name or output_dir.name}] trial {trial.number}: valid macro-F1={metrics['macro_f1']:.5f}, epoch={metrics['epoch']}", flush=True)
        return float(metrics["macro_f1"])

    finished = sum(trial.state in (optuna.trial.TrialState.COMPLETE, optuna.trial.TrialState.PRUNED) for trial in study.trials)
    if finished < n_trials:
        study.optimize(objective, n_trials=n_trials - finished, n_jobs=1, gc_after_trial=True)
    rows = []
    for trial in study.trials:
        rows.append({"number": trial.number, "state": trial.state.name, "value": trial.value,
                     "best_epoch": trial.user_attrs.get("best_epoch"), "training_seed": trial.user_attrs.get("training_seed"),
                     **trial.params})
    _csv(output_dir / "study_trials.csv", rows, ["number", "state", "value", "best_epoch", "training_seed", "lr", "batch_size", "weight_decay", "dropout", "hidden_dim", "class_weight_power"])
    best = study.best_trial
    result = {"study_name": study_name, "study_database": str(output_dir / "study.sqlite3"),
              "best_trial": best.number, "best_params": best.params, "best_validation_macro_f1": float(best.value),
              "best_epoch": int(best.user_attrs["best_epoch"]), "training_seed": int(best.user_attrs["training_seed"]),
              "data_signature": signature, "requested_trials": n_trials,
              "completed_trials": sum(t.state == optuna.trial.TrialState.COMPLETE for t in study.trials),
              "pruned_trials": sum(t.state == optuna.trial.TrialState.PRUNED for t in study.trials),
              "selection_split": "valid", "settings": settings}
    _json(output_dir / "best_params.json", result)
    return result


def cross_validate_head(train_features: np.ndarray, train_records: Sequence[Mapping[str, Any]],
                        params: Mapping[str, Any], output_dir: str | Path, *, device: str = "cpu",
                        seed: int = 42, cv_folds: int = 5, epochs: int = 12) -> dict[str, Any]:
    """Conditional, grouped CV with fixed parameters/epochs, not nested tuning."""
    _validate_records(train_records, "train")
    train_features = _validate_features(train_features, len(train_records))
    targets = labels_from_records(train_records)
    groups = np.asarray([str(record["group_id"]) for record in train_records])
    if not 2 <= cv_folds <= len(set(groups)):
        raise ValueError("CV needs at least two folds and enough independent saved groups.")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    splitter = StratifiedGroupKFold(n_splits=cv_folds, shuffle=True, random_state=seed)
    fold_indices = np.zeros(len(targets), dtype=np.int64)
    oof = np.zeros((len(targets), len(CLASS_NAMES)), dtype=np.float32)
    checkpoint_paths, metrics_rows = [], []
    for fold, (fit, held) in enumerate(splitter.split(np.zeros(len(targets)), targets, groups), start=1):
        if set(groups[fit]) & set(groups[held]):
            raise AssertionError("A saved capture group crosses a CV fold.")
        fold_dir = output_dir / f"fold_{fold}"
        head, history, metrics = train_head(train_features[fit], targets[fit], train_features[held], targets[held], params,
                                            device=device, seed=seed + 1000 + fold, max_epochs=epochs,
                                            select_best=False, output_dir=fold_dir)
        assert metrics["epoch"] == epochs
        checkpoint_path = fold_dir / "checkpoint.pt"
        save_head_checkpoint(checkpoint_path, head, {"stage": "grouped_cv", "fold": fold, "epochs": epochs,
                             "train_groups": len(set(groups[fit])), "held_out_groups": len(set(groups[held])),
                             "training_seed": seed + 1000 + fold, "normalization_fit": "fold_train_only"})
        checkpoint_paths.append(str(checkpoint_path.resolve()))
        oof[held] = predict_head(head, train_features[held], device=device)
        fold_indices[held] = fold
        _json(fold_dir / "metrics.json", metrics)
        metrics_rows.append({"fold": fold, "train_samples": len(fit), "validation_samples": len(held),
                             "train_groups": len(set(groups[fit])), "validation_groups": len(set(groups[held])), "epochs": epochs,
                             **{key: metrics[key] for key in ("macro_f1", "accuracy", "balanced_accuracy", "weighted_f1")}})
        print(f"[CV] fold {fold}/{cv_folds}: macro-F1={metrics['macro_f1']:.5f}", flush=True)
        del head
    if np.any(fold_indices == 0):
        raise AssertionError("Every training row must be held out in exactly one CV fold.")
    assignments = [{"row_index": i, "fold": int(fold_indices[i]), "split": "train", "group_id": record["group_id"],
                    "image_id": record.get("image_id"), "annotation_id": record.get("annotation_id"),
                    "crop_id": record.get("crop_id"), "label": CLASS_NAMES[targets[i]]} for i, record in enumerate(train_records)]
    _csv(output_dir / "fold_assignments.csv", assignments)
    _csv(output_dir / "fold_metrics.csv", metrics_rows)
    np.save(output_dir / "oof_probabilities.npy", oof, allow_pickle=False)
    oof_metrics = evaluate_predictions(targets, oof)
    summary = {"method": "StratifiedGroupKFold", "n_splits": cv_folds, "seed": seed, "split": "train",
               "fixed_epochs": epochs, "fixed_params": dict(params), "folds": metrics_rows,
               "macro_f1_mean": float(np.mean([r["macro_f1"] for r in metrics_rows])),
               "macro_f1_std": float(np.std([r["macro_f1"] for r in metrics_rows], ddof=1)),
               "oof_metrics": oof_metrics, "cv_checkpoint_paths": checkpoint_paths,
               "normalization": "fit separately on each fold training subset",
               "interpretation": "CV conditional on hyperparameters and epoch budget selected on the separate fixed validation split; not nested CV."}
    _json(output_dir / "summary.json", summary)
    return summary


def run_head_study(train_features: np.ndarray, train_records: Sequence[Mapping[str, Any]],
                   valid_features: np.ndarray, valid_records: Sequence[Mapping[str, Any]],
                   output_dir: str | Path, device: str = "cuda", seed: int = 42,
                   n_trials: int = 12, max_epochs: int = 12, cv_folds: int = 5,
                   model_name: str | None = None) -> dict[str, Any]:
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    tuning = tune_head(train_features, train_records, valid_features, valid_records, output_dir / "hyperband",
                       device=device, seed=seed, n_trials=n_trials, max_epochs=max_epochs, model_name=model_name)
    targets, valid_targets = labels_from_records(train_records), labels_from_records(valid_records)
    head, history, metrics = train_head(train_features, targets, valid_features, valid_targets, tuning["best_params"],
                                        device=device, seed=tuning["training_seed"], max_epochs=max_epochs,
                                        output_dir=output_dir)
    if metrics["epoch"] != tuning["best_epoch"] or not np.isclose(metrics["macro_f1"], tuning["best_validation_macro_f1"], atol=1e-6):
        raise RuntimeError("Selected trial could not be reproduced; inspect the device and deterministic settings.")
    checkpoint_path = output_dir / "checkpoint.pt"
    save_head_checkpoint(checkpoint_path, head, {"model_name": model_name, "stage": "frozen_backbone_transfer_learning",
                         "best_epoch": metrics["epoch"], "training_seed": tuning["training_seed"], "validation_metrics": metrics,
                         "normalization_fit": "train_only", "class_weights_fit": "train_only"})
    probabilities = predict_head(head, valid_features, device=device)
    validation_metrics = save_evaluation(output_dir / "validation", valid_records, probabilities, model_name=model_name,
                                        metadata={"strategy": "frozen_backbone_learned_head", "checkpoint_selection": "valid_macro_f1", "best_epoch": metrics["epoch"]})
    del head
    cv = cross_validate_head(train_features, train_records, tuning["best_params"], output_dir / "cross_validation",
                             device=device, seed=seed, cv_folds=cv_folds, epochs=int(metrics["epoch"]))
    result = {"model_name": model_name, "training_strategy": "frozen_backbone_transfer_learning", "seed": seed,
              "checkpoint_path": str(checkpoint_path), "model_state_path": str(checkpoint_path),
              "feature_dim": int(train_features.shape[1]), "class_labels": list(CLASS_NAMES),
              "best_params": tuning["best_params"], "best_epoch": int(metrics["epoch"]),
              "training_seed": tuning["training_seed"], "validation_metrics": validation_metrics,
              "cv_summary": cv, "cv_checkpoint_paths": cv["cv_checkpoint_paths"], "hyperband": tuning,
              "training_crops": len(train_records), "validation_crops": len(valid_records), "test_accessed": False}
    _json(output_dir / "study_summary.json", result)
    return result
