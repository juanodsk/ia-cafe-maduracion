"""Run the five frozen-backbone transfer-learning experiments on the saved split.

Feature extraction is batched; only small numeric embeddings are cached on disk.
Hyperband and grouped CV fit new classifiers without refitting the backbone.
The test partition is opened only by the explicit final-evaluation stage.
"""
from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("TORCH_HOME", str(ROOT / ".cache" / "torch"))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache" / "matplotlib"))
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from .data import CLASS_NAMES, CoffeeFruitDataset, load_records, prepare_crop_manifests
from .evaluation import save_evaluation
from .ensembles import ensemble_predictions
from .models import MODEL_NAMES, CNN_NAMES, TRANSFORMER_NAMES, build_model, get_feature_dim, get_model_spec, get_transforms, set_head


def json_write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def digest_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def seed_all(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.set_num_threads(4)


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    torch.set_num_threads(1)


def record_digest(records: list[dict]) -> str:
    serialized = json.dumps([
        [r["split"], r["image_id"], r["annotation_id"], r["group_id"], r["label"],
         r["parent_sha256"], r["crop_left"], r["crop_top"], r["crop_right"], r["crop_bottom"]]
        for r in records
    ], separators=(",", ":"))
    return hashlib.sha256(serialized.encode()).hexdigest()


def extract_features(name: str, split: str, records: list[dict], dataset_root: Path,
                     run_dir: Path, device: str, batch_size: int, workers: int) -> np.ndarray:
    cache_dir = run_dir / "features" / name
    cache_dir.mkdir(parents=True, exist_ok=True)
    feature_path = cache_dir / f"{split}.npy"
    meta_path = cache_dir / f"{split}.json"
    expected = {
        "model": get_model_spec(name), "records_sha256": record_digest(records),
        "torch": torch.__version__, "rows": len(records), "columns": get_feature_dim(name),
        "preprocessing": "official_weights_evaluation_preset; no new augmentation",
        "dtype": "float32", "autocast": "float16" if device == "cuda" else "disabled",
    }
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta.get("configuration") != expected or not feature_path.exists():
            raise ValueError(f"Incompatible feature cache: {cache_dir}. Use a new run directory.")
        if digest_file(feature_path) != meta.get("feature_file_sha256"):
            raise ValueError(f"Feature cache checksum mismatch: {feature_path}.")
        features = np.load(feature_path, mmap_mode="r")
        if features.shape != (len(records), get_feature_dim(name)):
            raise ValueError("Feature cache has an invalid shape.")
        print(f"[{name}] reuse {split} features: {features.shape}", flush=True)
        return features
    print(f"[{name}] extracting {split}: {len(records)} fruit crops", flush=True)
    seed_all(42)
    model = build_model(name, pretrained=True, trainable_backbone=False)
    set_head(model, name, nn.Identity())
    model.eval().to(device)
    dataset = CoffeeFruitDataset(records, dataset_root, get_transforms(name, train=False), cache_size=8)
    generator = torch.Generator().manual_seed(42)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=workers,
                        pin_memory=device == "cuda", worker_init_fn=seed_worker,
                        generator=generator, persistent_workers=workers > 0)
    features = np.lib.format.open_memmap(feature_path, mode="w+", dtype="float32",
                                        shape=(len(records), get_feature_dim(name)))
    offset = 0
    started = time.monotonic()
    with torch.inference_mode():
        for batch, _ in loader:
            batch = batch.to(device, non_blocking=device == "cuda")
            with torch.autocast(device_type=device, dtype=torch.float16, enabled=device == "cuda"):
                values = model(batch)
            values = values.float().cpu().numpy()
            if not np.isfinite(values).all():
                raise ValueError(f"Nonfinite embeddings in {name}/{split}.")
            features[offset:offset + len(values)] = values
            offset += len(values)
            if offset == len(records) or (offset // batch_size) % 50 == 0:
                elapsed = time.monotonic() - started
                print(f"[{name}/{split}] {offset}/{len(records)} crops; {offset / max(elapsed, .001):.1f} crops/s", flush=True)
    if offset != len(records):
        raise RuntimeError("Feature extraction did not consume every record.")
    features.flush()
    del features, loader, model
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
    json_write(meta_path, {"configuration": expected, "elapsed_seconds": time.monotonic() - started,
                           "feature_file_sha256": digest_file(feature_path)})
    return np.load(feature_path, mmap_mode="r")


def write_comparison(run_dir: Path, results: dict[str, dict]) -> None:
    rows = []
    for name, result in results.items():
        metrics = result["validation_metrics"]
        cv = result.get("cv_summary", {})
        rows.append({"model": name, "family": get_model_spec(name)["family"],
                     "valid_accuracy": metrics["accuracy"], "valid_macro_f1": metrics["macro_f1"],
                     "valid_balanced_accuracy": metrics["balanced_accuracy"],
                     "cv_macro_f1_mean": cv.get("macro_f1_mean", ""),
                     "cv_macro_f1_std": cv.get("macro_f1_std", ""),
                     "best_epoch": result.get("best_epoch", "")})
    rows.sort(key=lambda row: row["valid_macro_f1"], reverse=True)
    if rows:
        with (run_dir / "comparacion_validacion.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def train_models(args: argparse.Namespace) -> None:
    from .head_training import run_head_study, load_head_checkpoint, predict_head
    dataset_root, run_dir = args.dataset.resolve(), args.output.resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    configuration = {
        "phase": "frozen_backbone_transfer_learning", "seed": 42,
        "models": list(args.models), "class_labels": list(CLASS_NAMES),
        "dataset_manifest_sha256": digest_file(dataset_root / "csv" / "imagenes.csv"),
        "annotation_manifest_sha256": digest_file(dataset_root / "csv" / "anotaciones.csv"),
        "n_trials": args.trials, "max_epochs": args.epochs, "cv_folds": args.folds,
        "selection_metric": "validation_macro_f1", "test_reserved": True,
        "torch": torch.__version__, "python": sys.version,
        "device": args.device, "gpu": torch.cuda.get_device_name(0) if args.device == "cuda" else None,
    }
    config_path = run_dir / "configuracion.json"
    if config_path.exists() and json.loads(config_path.read_text(encoding="utf-8")) != configuration:
        raise ValueError("Run configuration differs from the saved experiment. Select a new output directory.")
    json_write(config_path, configuration)
    json_write(run_dir / "estado.json", {"stage": "preparing_manifests", "test_reserved": True})
    prepare_crop_manifests(dataset_root, include_test=False)
    train_records, valid_records = load_records(dataset_root, "train"), load_records(dataset_root, "valid")
    results = {}
    for name in args.models:
        model_dir = run_dir / name
        result_path = model_dir / "resultado.json"
        if result_path.exists():
            results[name] = json.loads(result_path.read_text(encoding="utf-8"))
            print(f"[{name}] completed result restored", flush=True)
            continue
        json_write(run_dir / "estado.json", {"stage": "feature_extraction", "model": name, "completed_models": list(results), "test_reserved": True})
        batch_size = min(args.batch_size, 32) if name == "vit_b_16" else args.batch_size
        train_features = extract_features(name, "train", train_records, dataset_root, run_dir, args.device, batch_size, args.workers)
        valid_features = extract_features(name, "valid", valid_records, dataset_root, run_dir, args.device, batch_size, args.workers)
        json_write(run_dir / "estado.json", {"stage": "hyperband_and_grouped_cv", "model": name, "completed_models": list(results), "test_reserved": True})
        result = run_head_study(train_features, train_records, valid_features, valid_records, model_dir,
                                device=args.device, seed=42, n_trials=args.trials, max_epochs=args.epochs,
                                cv_folds=args.folds, model_name=name)
        # Reevaluate the saved classifier in the exact original row order.
        head = load_head_checkpoint(result["checkpoint_path"], device=args.device)
        probabilities = predict_head(head, valid_features, device=args.device)
        result["validation_metrics"] = save_evaluation(model_dir / "validacion", valid_records, probabilities,
            model_name=name, metadata={"model_spec": get_model_spec(name), "training_scope": "classifier_head_only"})
        result["model_spec"] = get_model_spec(name)
        json_write(result_path, result)
        results[name] = result
        write_comparison(run_dir, results)
        del head, train_features, valid_features
        gc.collect()
        if args.device == "cuda": torch.cuda.empty_cache()
        print(f"[{name}] validation macro F1: {result['validation_metrics']['macro_f1']:.4f}", flush=True)
    build_validation_ensembles(run_dir, results, valid_records, args.device)
    json_write(run_dir / "estado.json", {"stage": "validation_complete", "completed_models": list(results), "test_reserved": True})
    print("All five training/validation experiments completed; test remains reserved.", flush=True)


def build_validation_ensembles(run_dir: Path, results: dict, valid_records: list[dict], device: str) -> dict:
    from .head_training import load_head_checkpoint, predict_head
    if set(results) != set(MODEL_NAMES):
        return {}
    root = run_dir / "ensembles"
    root.mkdir(exist_ok=True)
    definitions = {
        "heterogeneo_cinco": {"models": list(MODEL_NAMES), "weights": [0.2] * 5},
        "familia_cnn": {"models": list(CNN_NAMES), "weights": [1 / 3] * 3},
        "familia_transformer": {"models": list(TRANSFORMER_NAMES), "weights": [0.5, 0.5]},
        "hibrido_familias": {"models": list(MODEL_NAMES), "weights": [1 / 6] * 3 + [1 / 4] * 2},
    }
    scores = {}
    for name, definition in definitions.items():
        paths = [run_dir / model / "validacion" / "predictions.csv" for model in definition["models"]]
        metric = ensemble_predictions(paths, root / name / "validacion", weights=definition["weights"], model_name=name)
        scores[name] = metric["macro_f1"]
    best_model = max(results, key=lambda model: results[model]["validation_metrics"]["macro_f1"])
    checkpoints = results[best_model].get("cv_checkpoint_paths", [])
    if checkpoints:
        features = np.load(run_dir / "features" / best_model / "valid.npy", mmap_mode="r")
        probabilities = []
        for checkpoint in checkpoints:
            head = load_head_checkpoint(checkpoint, device=device)
            probabilities.append(predict_head(head, features, device=device))
            del head
        metric = save_evaluation(root / "homogeneo_cinco_folds" / "validacion", valid_records, np.mean(probabilities, axis=0), model_name="homogeneo_cinco_folds")
        definitions["homogeneo_cinco_folds"] = {"models": [best_model], "cv_checkpoints": checkpoints}
        scores["homogeneo_cinco_folds"] = metric["macro_f1"]
    candidates = {**{name: result["validation_metrics"]["macro_f1"] for name, result in results.items()}, **scores}
    selection = {"criterion": "validation_macro_f1_only", "scores": candidates,
                 "selected_candidate": max(candidates, key=candidates.get), "ensemble_definitions": definitions,
                 "test_used_for_selection": False}
    json_write(run_dir / "seleccion_validacion.json", selection)
    return selection


def final_evaluation(args: argparse.Namespace) -> None:
    from .head_training import load_head_checkpoint, predict_head
    run_dir, dataset_root = args.output.resolve(), args.dataset.resolve()
    selection_path = run_dir / "seleccion_validacion.json"
    if not selection_path.exists():
        raise ValueError("Complete training/validation and save selection before opening test.")
    final_path = run_dir / "evaluacion_final.json"
    if final_path.exists():
        raise ValueError("Final evaluation already exists. Preserve it; do not retune using test results.")
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    json_write(run_dir / "estado.json", {"stage": "final_test_evaluation", "selection_locked": selection["selected_candidate"], "test_reserved": False})
    records = load_records(dataset_root, "test", allow_test=True)
    prepare_crop_manifests(dataset_root, include_test=True)
    metrics = {}
    for name in MODEL_NAMES:
        result = json.loads((run_dir / name / "resultado.json").read_text(encoding="utf-8"))
        batch_size = min(args.batch_size, 32) if name == "vit_b_16" else args.batch_size
        features = extract_features(name, "test", records, dataset_root, run_dir, args.device, batch_size, args.workers)
        head = load_head_checkpoint(result["checkpoint_path"], device=args.device)
        probabilities = predict_head(head, features, device=args.device)
        metrics[name] = save_evaluation(run_dir / name / "prueba", records, probabilities, model_name=name, allow_test=True)
        del head
    for name, definition in selection["ensemble_definitions"].items():
        if "cv_checkpoints" in definition:
            features = np.load(run_dir / "features" / definition["models"][0] / "test.npy", mmap_mode="r")
            probabilities = []
            for checkpoint in definition["cv_checkpoints"]:
                head = load_head_checkpoint(checkpoint, device=args.device)
                probabilities.append(predict_head(head, features, device=args.device))
                del head
            metrics[name] = save_evaluation(run_dir / "ensembles" / name / "prueba", records, np.mean(probabilities, axis=0), model_name=name, allow_test=True)
        else:
            paths = [run_dir / model / "prueba" / "predictions.csv" for model in definition["models"]]
            metrics[name] = ensemble_predictions(paths, run_dir / "ensembles" / name / "prueba", weights=definition["weights"], model_name=name, allow_test=True)
    result = {"selected_before_test": selection["selected_candidate"], "test_metrics": metrics,
              "test_crops": len(records), "training_scope": "frozen_ImageNet_backbones_and_trained_coffee_classifier_heads",
              "limitations": "Source v5 contains inherited augmentations; grouping cannot establish independence across unidentified physical plants/fruits."}
    json_write(final_path, result)
    rows = [{"candidate": name, **{key: values[key] for key in ("accuracy", "balanced_accuracy", "macro_f1", "weighted_f1")}} for name, values in metrics.items()]
    with (run_dir / "comparacion_prueba.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    json_write(run_dir / "estado.json", {"stage": "complete", "selected_before_test": selection["selected_candidate"], "test_evaluated": True})
    print(json.dumps({"selected_before_test": selection["selected_candidate"], "test_metrics": metrics[selection["selected_candidate"]]}, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("download", "train", "evaluate"))
    parser.add_argument("--dataset", type=Path, default=ROOT / "dataset_preparado")
    parser.add_argument("--output", type=Path, default=ROOT / "experimentos" / "transfer_01")
    parser.add_argument("--models", nargs="+", choices=MODEL_NAMES, default=list(MODEL_NAMES))
    parser.add_argument("--trials", type=int, default=12)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Fix the GPU environment or explicitly select --device cpu.")
    seed_all(42)
    if args.phase == "download":
        for name in args.models:
            print(f"Downloading/verifying pinned weights for {name}", flush=True)
            model = build_model(name, pretrained=True)
            del model; gc.collect()
        return
    if args.phase == "train": train_models(args)
    else: final_evaluation(args)


if __name__ == "__main__":
    main()
