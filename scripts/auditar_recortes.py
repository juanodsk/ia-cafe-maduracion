"""Audit real training/validation crops and Windows multiworker loading on CPU.

No pretrained weights are downloaded, no GPU is used and no test records are
opened. Selects representative parent images from every maturity class.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from torch.utils.data import DataLoader

from cafe_ml.data import CLASS_NAMES, CoffeeFruitDataset, load_records
from cafe_ml.models import MODEL_NAMES, get_transforms


def worker_init(worker_id: int) -> None:
    torch.set_num_threads(1)


def digest_records(records: list[dict]) -> str:
    payload = json.dumps(records, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def representatives(records: list[dict], per_class: int) -> list[dict]:
    selected = []
    for label in CLASS_NAMES:
        parents = set()
        for record in records:
            if record["label"] != label or record["image_id"] in parents:
                continue
            parents.add(record["image_id"])
            selected.append(record)
            if len(parents) == per_class:
                break
        if len(parents) != per_class:
            raise ValueError(f"Not enough distinct parent images for class {label}")
    return selected


def run(dataset_root: Path, workers: int, per_class: int) -> dict:
    torch.set_num_threads(2)
    result = {"status": "OK", "device": "cpu", "workers": workers,
              "test_records_opened": False, "partitions": {}, "official_transform_shapes": {}}
    selected = {}
    for split in ("train", "valid"):
        records = load_records(dataset_root, split)
        original_digest = digest_records(records)
        samples = representatives(records, per_class)
        selected[split] = samples
        dataset = CoffeeFruitDataset(samples, dataset_root, transform=None, cache_size=2)
        for index, record in enumerate(samples):
            crop, target = dataset[index]
            expected = (record["crop_right"] - record["crop_left"], record["crop_bottom"] - record["crop_top"])
            assert crop.mode == "RGB" and crop.size == expected and min(crop.size) > 0
            assert target == record["class_index"]
            assert record["split"] == split
        assert digest_records(records) == original_digest, "Crop loading mutated metadata"
        result["partitions"][split] = {
            "total_records": len(records), "audited_crops": len(samples),
            "parent_images": len({r["image_id"] for r in samples}),
            "labels": dict(Counter(r["label"] for r in samples)),
            "metadata_unchanged": True, "pil_rgb_and_bounds": "verified",
        }
    for model in MODEL_NAMES:
        dataset = CoffeeFruitDataset(selected["train"] + selected["valid"], dataset_root,
                                     transform=get_transforms(model, train=False), cache_size=2)
        for index in range(len(dataset)):
            tensor, target = dataset[index]
            assert tensor.shape == (3, 224, 224), (model, tensor.shape)
            assert tensor.dtype == torch.float32 and torch.isfinite(tensor).all()
            assert target == dataset.targets[index]
        result["official_transform_shapes"][model] = [3, 224, 224]
    # Use actual spawned Windows workers. The main guard makes this safe.
    for split, samples in selected.items():
        dataset = CoffeeFruitDataset(samples, dataset_root, get_transforms("resnet50", train=False))
        loader = DataLoader(dataset, batch_size=7, shuffle=False, num_workers=workers,
                            worker_init_fn=worker_init, persistent_workers=False, pin_memory=False)
        labels, batch_shapes = [], []
        for batch, targets in loader:
            assert batch.dtype == torch.float32 and batch.device.type == "cpu"
            assert batch.shape[1:] == (3, 224, 224) and torch.isfinite(batch).all()
            labels.extend(targets.tolist())
            batch_shapes.append(list(batch.shape))
        assert labels == dataset.targets, "Multiworker loader changed order or labels"
        result["partitions"][split]["multiworker_batch_shapes"] = batch_shapes
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=ROOT / "dataset_preparado")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--per-class", type=int, default=3)
    parser.add_argument("--output", type=Path, default=ROOT / "experimentos" / "auditoria_recortes.json")
    args = parser.parse_args()
    if args.workers < 0 or args.per_class < 1:
        parser.error("workers must be nonnegative and per-class must be positive")
    result = run(args.dataset.resolve(), args.workers, args.per_class)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
