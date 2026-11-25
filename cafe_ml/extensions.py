"""Run train-only GAN ablation, simulated FedAvg and real two-process CPU DDP.

Run this stage after the five baseline models and their validation selection.
Each phase preserves a fingerprint and artifacts for safe, auditable resumption.
This module never loads the reserved test partition.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("TORCH_HOME", str(ROOT / ".cache" / "torch"))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache" / "matplotlib"))
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch

from .data import CLASS_NAMES, load_records
from .evaluation import save_evaluation
from .experiment import digest_file, extract_features, json_write, record_digest
from .head_training import labels_from_records, load_head_checkpoint, predict_head, save_head_checkpoint, train_head
from .models import MODEL_NAMES


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _fingerprint(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _load_context(dataset: Path, run_dir: Path) -> dict[str, Any]:
    if (run_dir / "evaluacion_final.json").exists():
        raise ValueError("Test has already been evaluated. Do not run new training/selection in this experiment after seeing test results.")
    config = _read_json(run_dir / "configuracion.json")
    if set(config.get("models", [])) != set(MODEL_NAMES):
        raise ValueError("Extensions require the configured three CNNs and two Transformers.")
    if config.get("class_labels") != list(CLASS_NAMES):
        raise ValueError("Experiment class vocabulary differs from the project.")
    for key, name in (("dataset_manifest_sha256", "imagenes.csv"), ("annotation_manifest_sha256", "anotaciones.csv")):
        if digest_file(dataset / "csv" / name) != config[key]:
            raise ValueError("Dataset manifest changed after baseline training.")
    results = {}
    result_hashes = {}
    for name in MODEL_NAMES:
        path = run_dir / name / "resultado.json"
        if not path.is_file():
            raise ValueError(f"Complete all five baseline models before extensions; missing {name}/resultado.json.")
        result = _read_json(path)
        score = result.get("validation_metrics", {}).get("macro_f1")
        if not isinstance(score, (int, float)) or not np.isfinite(score):
            raise ValueError(f"Invalid saved validation score for {name}.")
        if result.get("test_accessed", False):
            raise ValueError("Baseline result reports test access before extension selection.")
        results[name] = result
        result_hashes[name] = digest_file(path)
    selection_path = run_dir / "seleccion_validacion.json"
    if not selection_path.is_file():
        raise ValueError("Complete and save baseline validation selection before extensions.")
    selection = _read_json(selection_path)
    if selection.get("test_used_for_selection") is not False:
        raise ValueError("Baseline selection must explicitly exclude test.")
    best_name = max(MODEL_NAMES, key=lambda name: results[name]["validation_metrics"]["macro_f1"])
    baseline = results[best_name]
    if baseline.get("training_strategy") != "frozen_backbone_transfer_learning":
        raise ValueError("Extensions expect frozen-backbone features and a learned head.")
    train_records, valid_records = load_records(dataset, "train"), load_records(dataset, "valid")
    features = {}
    feature_hashes = {}
    for split, records in (("train", train_records), ("valid", valid_records)):
        cache_dir = run_dir / "features" / best_name
        meta = _read_json(cache_dir / f"{split}.json")
        if meta.get("configuration", {}).get("records_sha256") != record_digest(records):
            raise ValueError(f"Cached {split} feature row identities do not match the current crop manifest.")
        array_path = cache_dir / f"{split}.npy"
        file_hash = digest_file(array_path)
        if file_hash != meta.get("feature_file_sha256"):
            raise ValueError(f"Cached {split} features changed after extraction.")
        values = np.load(array_path, mmap_mode="r", allow_pickle=False)
        if values.ndim != 2 or values.shape[0] != len(records) or values.shape[1] != int(baseline["feature_dim"]):
            raise ValueError(f"Invalid cached {split} feature matrix shape.")
        features[split] = values
        feature_hashes[split] = file_hash
    checkpoint = Path(baseline["checkpoint_path"])
    if not checkpoint.is_absolute():
        checkpoint = run_dir / best_name / checkpoint.name
    head = load_head_checkpoint(checkpoint, device="cpu")
    training_seed = int(baseline["training_seed"])
    best_epoch = int(baseline["best_epoch"])
    if head.checkpoint_metadata.get("training_seed") != training_seed:
        raise ValueError("Baseline checkpoint training seed differs from the reported selected trial.")
    if head.checkpoint_metadata.get("best_epoch") != best_epoch or head.params != baseline["best_params"]:
        raise ValueError("Baseline checkpoint epoch/parameters differ from the saved result.")
    del head
    identity = {"schema_version": 1, "dataset_manifest_sha256": config["dataset_manifest_sha256"],
                "annotation_manifest_sha256": config["annotation_manifest_sha256"],
                "baseline_result_sha256": result_hashes, "baseline_selection_sha256": digest_file(selection_path),
                "best_individual_model": best_name, "selection_rule": "highest_fixed_validation_macro_f1_individual_only",
                "baseline_checkpoint_sha256": digest_file(checkpoint), "feature_file_sha256": feature_hashes,
                "head_params": baseline["best_params"], "head_training_seed": training_seed,
                "head_epoch_budget": best_epoch, "class_labels": list(CLASS_NAMES), "test_used": False}
    return {"dataset": dataset, "run_dir": run_dir, "config": config, "results": results, "baseline": baseline,
            "model_name": best_name, "training_seed": training_seed, "best_epoch": best_epoch,
            "train_records": train_records, "valid_records": valid_records,
            "train_features": features["train"], "valid_features": features["valid"],
            "identity": identity, "fingerprint": _fingerprint(identity)}


def _summary(context: dict[str, Any]) -> dict[str, Any]:
    path = context["run_dir"] / "resumen_extensiones.json"
    if path.exists():
        summary = _read_json(path)
        if summary.get("baseline_fingerprint") != context["fingerprint"]:
            raise ValueError("Saved extensions belong to a different baseline/dataset. Use another experiment directory.")
        return summary
    baseline = context["baseline"]
    return {"schema_version": 1, "baseline_fingerprint": context["fingerprint"],
            "best_individual_model": context["model_name"], "selection_rule": "validation_macro_f1_only",
            "baseline_validation_metrics": baseline["validation_metrics"],
            "baseline_params": baseline["best_params"], "baseline_training_seed": context["training_seed"],
            "baseline_best_epoch": context["best_epoch"], "test_used": False,
            "baseline_selection_modified": False,
            "phases": {name: {"status": "pending"} for name in ("gan", "gan_ablation", "federated", "distributed")}}


def _status(context: dict[str, Any], phase: str, status: str, **details: Any) -> None:
    summary = _summary(context)
    summary["phases"][phase] = {"status": status, **details}
    json_write(context["run_dir"] / "resumen_extensiones.json", summary)
    json_write(context["run_dir"] / "estado_extensiones.json", {"phase": phase, "status": status,
               "best_individual_model": context["model_name"], "test_reserved": True, **details})


def _phase(
    context: dict[str, Any], name: str, settings: dict[str, Any], artifact_dir: Path,
    execute: Callable[[], dict[str, Any]],
) -> dict[str, Any]:
    request = {"baseline_fingerprint": context["fingerprint"], "phase": name, "settings": settings}
    fingerprint = _fingerprint(request)
    phase_dir = context["run_dir"] / "extensiones"
    phase_dir.mkdir(exist_ok=True)
    request_path = phase_dir / f"{name}_solicitud.json"
    result_path = phase_dir / f"{name}_completo.json"
    if request_path.exists() and _read_json(request_path) != request:
        raise ValueError(f"Saved {name} request has a different fingerprint/configuration.")
    if not request_path.exists():
        json_write(request_path, request)
    if result_path.exists():
        saved = _read_json(result_path)
        if saved.get("fingerprint") != fingerprint:
            raise ValueError(f"Completed {name} phase fingerprint differs from the current request.")
        for item in saved.get("artifacts", []):
            path = context["run_dir"] / item["relative_path"]
            if not path.is_file() or digest_file(path) != item["sha256"]:
                raise ValueError(f"Completed {name} artifact is missing or changed: {path}")
        _status(context, name, "complete", fingerprint=fingerprint, result=saved["result"], reused=True)
        print(f"[extensions/{name}] verified completed artifacts; reused", flush=True)
        return saved["result"]
    _status(context, name, "running", fingerprint=fingerprint, settings=settings)
    print(f"[extensions/{name}] starting; test remains reserved", flush=True)
    try:
        result = execute()
        artifacts = [{"relative_path": file.relative_to(context["run_dir"]).as_posix(), "sha256": digest_file(file)}
                     for file in sorted(artifact_dir.rglob("*")) if file.is_file() and not file.name.endswith(".tmp")]
        if not artifacts:
            raise RuntimeError(f"{name} returned without any saved artifacts.")
        payload = {"fingerprint": fingerprint, "request": request, "result": result, "artifacts": artifacts, "test_used": False}
        json_write(result_path, payload)
        _status(context, name, "complete", fingerprint=fingerprint, result=result, reused=False)
        return result
    except Exception as error:
        _status(context, name, "failed", fingerprint=fingerprint, error_type=type(error).__name__, error=str(error))
        raise


def _run_gan(context: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    from .gan import train_gan

    settings = {"device": args.device, "epochs": args.gan_epochs, "steps_per_epoch": args.gan_steps,
                "batch_size": args.gan_batch_size, "seed": 42, "synthetic_per_class": args.synthetic_per_class,
                "workers": args.workers, "train_only": True}
    directory = context["run_dir"] / "gan"
    return _phase(context, "gan", settings, directory, lambda: train_gan(
        context["dataset"], directory, device=args.device, epochs=args.gan_epochs,
        steps_per_epoch=args.gan_steps, batch_size=args.gan_batch_size, seed=42,
        synthetic_per_class=args.synthetic_per_class, workers=args.workers,
    ))


def _run_gan_ablation(context: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    from .gan import load_synthetic_records

    gan_dir = context["run_dir"] / "gan"
    gan_result_path = gan_dir / "resultado.json"
    if not gan_result_path.is_file():
        raise ValueError("Train and finish the GAN phase before its augmentation ablation.")
    gan_result = _read_json(gan_result_path)
    if gan_result.get("stage") != "complete":
        raise ValueError("GAN generation is not complete.")
    records = load_synthetic_records(gan_dir)
    if any(record.get("split") != "train" or not record.get("synthetic") for record in records):
        raise ValueError("Synthetic rows must be explicitly restricted to train.")
    if {r["group_id"] for r in records} & {r["group_id"] for r in context["valid_records"]}:
        raise ValueError("A synthetic group overlaps validation.")
    settings = {"device": args.device, "model_name": context["model_name"],
                "params": context["baseline"]["best_params"], "seed": context["training_seed"],
                "epochs": context["best_epoch"], "select_best": False,
                "synthetic_records_sha256": record_digest(records), "gan_result_sha256": digest_file(gan_result_path),
                "validation_records_sha256": record_digest(context["valid_records"]),
                "cross_validation": False, "reason_no_cv": "generator_fitted_on_all_train"}
    directory = context["run_dir"] / "gan_ablation"

    def execute() -> dict[str, Any]:
        synthetic_features = extract_features(context["model_name"], "synthetic_train", records, gan_dir,
            context["run_dir"], args.device, min(64, args.feature_batch_size), args.workers)
        augmented = np.concatenate([context["train_features"], synthetic_features]).astype(np.float32, copy=False)
        targets = np.concatenate([labels_from_records(context["train_records"]), labels_from_records(records)])
        head, _, fit_metrics = train_head(augmented, targets, context["valid_features"],
            labels_from_records(context["valid_records"]), context["baseline"]["best_params"], device=args.device,
            seed=context["training_seed"], max_epochs=context["best_epoch"], select_best=False, output_dir=directory)
        if fit_metrics["seed"] != context["training_seed"] or fit_metrics["epoch"] != context["best_epoch"]:
            raise AssertionError("GAN ablation must match the baseline checkpoint's seed and epoch budget.")
        checkpoint = directory / "checkpoint.pt"
        save_head_checkpoint(checkpoint, head, {"stage": "gan_augmentation_ablation", "model_name": context["model_name"],
            "training_seed": context["training_seed"], "baseline_training_seed": context["training_seed"],
            "fixed_epochs": context["best_epoch"], "baseline_best_epoch": context["best_epoch"],
            "synthetic_use": "train_only", "baseline_fingerprint": context["fingerprint"], "cross_validation": False,
            "normalization_fit": "real_and_synthetic_train_only", "class_weights_fit": "real_and_synthetic_train_only"})
        probabilities = predict_head(head, context["valid_features"], device=args.device)
        metrics = save_evaluation(directory / "validacion", context["valid_records"], probabilities,
            model_name=context["model_name"] + "_gan_augmentation",
            metadata={"training_scope": "classifier_head_only", "synthetic_use": "train_only",
                      "fixed_epochs": context["best_epoch"], "training_seed": context["training_seed"],
                      "validation_used_for_checkpoint_selection": False})
        baseline_metrics = context["baseline"]["validation_metrics"]
        result = {"stage": "complete", "method": "train_only_conditional_gan_augmentation_ablation",
                  "model_name": context["model_name"], "baseline_validation_metrics": baseline_metrics,
                  "validation_metrics": metrics, "macro_f1_change": metrics["macro_f1"] - baseline_metrics["macro_f1"],
                  "params": context["baseline"]["best_params"], "training_seed": context["training_seed"],
                  "baseline_training_seed": context["training_seed"], "fixed_epochs": context["best_epoch"],
                  "baseline_best_epoch": context["best_epoch"], "checkpoint_path": str(checkpoint.resolve()),
                  "real_training_crops": len(context["train_records"]), "synthetic_training_crops": len(records),
                  "validation_crops": len(context["valid_records"]), "baseline_fingerprint": context["fingerprint"],
                  "cross_validation": False, "reason_no_cv": "GAN fitted on all train; synthetic images would contaminate held-out train folds",
                  "test_used": False, "selection_modified": False}
        json_write(directory / "resultado.json", result)
        del head, augmented, synthetic_features
        gc.collect()
        if args.device == "cuda":
            torch.cuda.empty_cache()
        return result

    return _phase(context, "gan_ablation", settings, directory, execute)


def _run_federated(context: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    from .federated import run_federated_experiment

    settings = {"params": context["baseline"]["best_params"], "n_clients": args.fed_clients,
                "rounds": args.fed_rounds, "local_epochs": args.fed_local_epochs,
                "seed": context["training_seed"], "device": "cpu", "simulation": True}
    directory = context["run_dir"] / "federated"
    return _phase(context, "federated", settings, directory, lambda: run_federated_experiment(
        context["train_features"], context["train_records"], context["valid_features"], context["valid_records"], directory,
        params=context["baseline"]["best_params"], n_clients=args.fed_clients, rounds=args.fed_rounds,
        local_epochs=args.fed_local_epochs, seed=context["training_seed"], device="cpu",
    ))


def _run_distributed(context: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    # The Windows CUDA 2.9.1 wheel has no usable Gloo transport on this host.
    # Keep that training environment intact and use the verified CPU wheel.
    cpu_python = ROOT / ".venv_ddp" / "Scripts" / "python.exe"
    if os.name == "nt" and cpu_python.is_file() and Path(sys.executable).resolve() != cpu_python.resolve():
        subprocess.run([str(cpu_python), "-u", "-m", "cafe_ml.extensions", "--phase", "distributed",
                        "--dataset", str(context["dataset"]), "--run", str(context["run_dir"]),
                        "--ddp-processes", str(args.ddp_processes), "--ddp-epochs", str(args.ddp_epochs)],
                       cwd=ROOT, check=True)
        return _read_json(context["run_dir"] / "distributed" / "summary.json")
    from .distributed import run_distributed_experiment

    settings = {"params": context["baseline"]["best_params"], "world_size": args.ddp_processes,
                "epochs": args.ddp_epochs, "seed": context["training_seed"], "device": "cpu", "backend": "gloo"}
    directory = context["run_dir"] / "distributed"
    return _phase(context, "distributed", settings, directory, lambda: run_distributed_experiment(
        context["train_features"], context["train_records"], context["valid_features"], context["valid_records"], directory,
        params=context["baseline"]["best_params"], world_size=args.ddp_processes, epochs=args.ddp_epochs,
        seed=context["training_seed"],
    ))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=ROOT / "dataset_preparado")
    parser.add_argument("--run", type=Path, default=ROOT / "experimentos" / "transfer_01")
    parser.add_argument("--phase", choices=("all", "gan", "gan-ablation", "federated", "distributed"), default="all")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda", help="GAN and ablation device; FedAvg/DDP run on CPU.")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--feature-batch-size", type=int, default=64)
    parser.add_argument("--gan-epochs", type=int, default=5)
    parser.add_argument("--gan-steps", type=int, default=80)
    parser.add_argument("--gan-batch-size", type=int, default=64)
    parser.add_argument("--synthetic-per-class", type=int, default=256)
    parser.add_argument("--fed-clients", type=int, default=3)
    parser.add_argument("--fed-rounds", type=int, default=5)
    parser.add_argument("--fed-local-epochs", type=int, default=2)
    parser.add_argument("--ddp-processes", type=int, default=2)
    parser.add_argument("--ddp-epochs", type=int, default=2)
    args = parser.parse_args()
    if min(args.feature_batch_size, args.gan_epochs, args.gan_steps, args.gan_batch_size,
           args.synthetic_per_class, args.fed_clients, args.fed_rounds, args.fed_local_epochs,
           args.ddp_processes, args.ddp_epochs) < 1 or args.workers < 0:
        parser.error("Training budgets must be positive and workers nonnegative.")
    if args.phase in ("all", "gan", "gan-ablation") and args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; explicitly select --device cpu for a CPU pilot.")
    torch.set_num_threads(4)
    context = _load_context(args.dataset.resolve(), args.run.resolve())
    json_write(context["run_dir"] / "resumen_extensiones.json", _summary(context))
    print(f"[extensions] best individual by validation: {context['model_name']}; seed {context['training_seed']}; epoch {context['best_epoch']}", flush=True)
    if args.phase in ("all", "gan"):
        _run_gan(context, args)
    if args.phase in ("all", "gan", "gan-ablation"):
        _run_gan_ablation(context, args)
    if args.phase in ("all", "federated"):
        _run_federated(context, args)
    if args.phase in ("all", "distributed"):
        _run_distributed(context, args)
    summary = _summary(context)
    print(json.dumps({"best_individual_model": summary["best_individual_model"], "test_used": False,
        "phases": {key: {"status": phase["status"], "valid_macro_f1": phase.get("result", {}).get("validation_metrics", {}).get("macro_f1")}
                   for key, phase in summary["phases"].items()}}, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
