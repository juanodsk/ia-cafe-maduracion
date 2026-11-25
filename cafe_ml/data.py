"""Lazy fruit crops from the fixed COCO image partitions.

No images are loaded when records/manifests are read. Each item opens at most one
parent image, crops its annotated box and applies the caller's transform. The
small image cache is per DataLoader worker and never holds the full dataset.
"""

from __future__ import annotations

import csv
import math
from collections import Counter, OrderedDict
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence


CLASS_NAMES = ("dry", "overripe", "ripe", "semi_ripe", "unripe")
CLASS_TO_INDEX = {label: index for index, label in enumerate(CLASS_NAMES)}
SPLITS = ("train", "valid", "test")


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"CSV without headers: {path}")
        return list(reader)


@lru_cache(maxsize=8192)
def _image_path(dataset_root: Path, relative_path: str) -> Path:
    """Resolve only paths inside the selected dataset, including on Windows."""
    target = (dataset_root / relative_path).resolve()
    if not target.is_relative_to(dataset_root.resolve()):
        raise ValueError(f"Image path escapes dataset root: {relative_path!r}")
    return target


def crop_bounds(
    bbox: Sequence[float], image_width: int, image_height: int
) -> tuple[int, int, int, int]:
    """COCO xywh -> clipped PIL xyxy box, including fractional boundary pixels.

    Boxes fully outside the image or with nonpositive dimensions are rejected.
    Out-of-bounds portions are clipped rather than filled with black padding.
    """
    if len(bbox) != 4 or not all(math.isfinite(float(value)) for value in bbox):
        raise ValueError(f"Invalid COCO box: {bbox!r}")
    x, y, width, height = map(float, bbox)
    if width <= 0 or height <= 0 or image_width <= 0 or image_height <= 0:
        raise ValueError(f"Nonpositive box/image dimensions: {bbox!r}")
    left = max(0, min(image_width, math.floor(x)))
    top = max(0, min(image_height, math.floor(y)))
    right = max(0, min(image_width, math.ceil(x + width)))
    bottom = max(0, min(image_height, math.ceil(y + height)))
    if right <= left or bottom <= top:
        raise ValueError(f"Box does not intersect the parent image: {bbox!r}")
    return left, top, right, bottom


def validate_split_integrity(dataset_root: str | Path) -> dict[str, int]:
    """Check image-level group/hash exclusivity without consuming test labels."""
    dataset_root = Path(dataset_root).resolve()
    images = _read_csv(dataset_root / "csv" / "imagenes.csv")
    seen: dict[str, dict[str, str]] = {"group_id": {}, "sha256": {}, "source_name": {}}
    identifiers: set[tuple[str, str]] = set()
    counts: Counter[str] = Counter()
    for image in images:
        split = image["split"]
        if split not in SPLITS:
            raise ValueError(f"Unknown partition in image manifest: {split!r}")
        key = (split, image["image_id"])
        if key in identifiers:
            raise ValueError(f"Repeated image ID in {split}: {image['image_id']}")
        identifiers.add(key)
        for field, assignments in seen.items():
            value = image.get(field, "")
            if not value:
                raise ValueError(f"Image {key} has no {field} provenance")
            if value in assignments and assignments[value] != split:
                raise ValueError(f"Data leakage: {field} {value!r} crosses partitions")
            assignments[value] = split
        _image_path(dataset_root, image["relative_path"])
        counts[split] += 1
    if set(counts) != set(SPLITS):
        raise ValueError("Expected nonempty train, valid and test image partitions")
    return dict(counts)


def load_records(
    dataset_root: str | Path, split: str, allow_test: bool = False
) -> list[dict[str, Any]]:
    """Return audited fruit records inheriting the saved parent image split.

    ``test`` is deliberately opt-in for final evaluation. This function never
    repartitions fruit crops and never balances validation/test by resampling.
    """
    if split not in SPLITS:
        raise ValueError(f"Unknown split {split!r}; expected one of {SPLITS}")
    if split == "test" and not allow_test:
        raise ValueError("The test set is reserved; use allow_test=True only for final evaluation")
    dataset_root = Path(dataset_root).resolve()
    validate_split_integrity(dataset_root)
    parents = {
        row["image_id"]: row
        for row in _read_csv(dataset_root / "csv" / "imagenes.csv")
        if row["split"] == split
    }
    records: list[dict[str, Any]] = []
    seen_annotations: set[int] = set()
    with (dataset_root / "csv" / "anotaciones.csv").open(
        "r", encoding="utf-8-sig", newline=""
    ) as handle:
        for row in csv.DictReader(handle):
            if row["split"] != split:
                continue
            annotation_id = int(row["annotation_id"])
            if annotation_id in seen_annotations:
                raise ValueError(f"Repeated annotation ID in {split}: {annotation_id}")
            seen_annotations.add(annotation_id)
            parent = parents.get(row["image_id"])
            if parent is None:
                raise ValueError(f"Annotation {annotation_id} references a missing parent image")
            if row["group_id"] != parent["group_id"]:
                raise ValueError(f"Annotation {annotation_id} has a different group than its image")
            if row["relative_image_path"] != parent["relative_path"]:
                raise ValueError(f"Annotation {annotation_id} has a different path than its image")
            label = row["label"]
            if label not in CLASS_TO_INDEX:
                raise ValueError(f"Unknown maturity label: {label!r}")
            if int(row["category_id"]) != CLASS_TO_INDEX[label] + 1:
                raise ValueError(f"COCO category/label mismatch for annotation {annotation_id}")
            width, height = int(parent["width"]), int(parent["height"])
            bbox = tuple(float(row[name]) for name in ("bbox_x", "bbox_y", "bbox_width", "bbox_height"))
            bounds = crop_bounds(bbox, width, height)
            _image_path(dataset_root, row["relative_image_path"])
            record: dict[str, Any] = dict(row)
            for name in ("annotation_id", "image_id", "category_id", "source_image_id", "source_annotation_id"):
                record[name] = int(row[name])
            for name, value in zip(("bbox_x", "bbox_y", "bbox_width", "bbox_height"), bbox):
                record[name] = value
            record.update(
                crop_id=f"{split}:{row['image_id']}:{annotation_id}",
                class_index=CLASS_TO_INDEX[label], image_width=width, image_height=height,
                crop_left=bounds[0], crop_top=bounds[1], crop_right=bounds[2], crop_bottom=bounds[3],
                parent_sha256=parent["sha256"], capture_key=parent["capture_key"],
            )
            records.append(record)
    if not records:
        raise ValueError(f"No fruit records in split {split!r}")
    parent_counts = Counter(str(record["image_id"]) for record in records)
    for image_id, parent in parents.items():
        if parent_counts[image_id] != int(parent["annotation_count"]):
            raise ValueError(f"Annotation count mismatch for {split} image {image_id}")
    return records


def prepare_crop_manifests(
    dataset_root: str | Path,
    output_dir: str | Path | None = None,
    *,
    include_test: bool = False,
) -> dict[str, dict[str, Any]]:
    """Save deterministic crop metadata; create no duplicated image files.

    The saved ``recortes_<split>.csv`` has one row per annotated fruit. A parent
    image may yield many rows, always sharing its fixed split and group.
    """
    dataset_root = Path(dataset_root).resolve()
    output_dir = Path(output_dir) if output_dir is not None else dataset_root / "csv"
    output_dir.mkdir(parents=True, exist_ok=True)
    results: dict[str, dict[str, Any]] = {}
    for split in SPLITS if include_test else SPLITS[:2]:
        records = load_records(dataset_root, split, allow_test=include_test)
        file = output_dir / f"recortes_{split}.csv"
        with file.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(records[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(records)
        results[split] = {
            "path": str(file), "crops": len(records),
            "images": len({record["image_id"] for record in records}),
            "groups": len({record["group_id"] for record in records}),
            "class_counts": {label: sum(r["label"] == label for r in records) for label in CLASS_NAMES},
            "clipped_boxes": sum(
                r["bbox_x"] < 0 or r["bbox_y"] < 0 or
                r["bbox_x"] + r["bbox_width"] > r["image_width"] or
                r["bbox_y"] + r["bbox_height"] > r["image_height"]
                for r in records
            ),
        }
    return results


class CoffeeFruitDataset:
    """Map-style PyTorch-compatible dataset with a bounded per-worker PIL cache.

    Supply a torchvision transform to produce tensors. Without a transform the
    returned crop is a PIL RGB image, useful for auditing. Corrupt/missing images
    raise a contextual error rather than silently changing the evaluation set.
    """

    def __init__(
        self,
        records: Iterable[dict[str, Any]],
        dataset_root: str | Path,
        transform: Callable | None = None,
        cache_size: int = 4,
    ) -> None:
        if cache_size < 0:
            raise ValueError("cache_size must be nonnegative")
        self.records = list(records)
        self.dataset_root = Path(dataset_root).resolve()
        self.transform = transform
        self.cache_size = cache_size
        self._cache: OrderedDict[str, Any] = OrderedDict()
        self.targets = [int(record["class_index"]) for record in self.records]
        self.groups = [record["group_id"] for record in self.records]
        self.classes = list(CLASS_NAMES)

    def __len__(self) -> int:
        return len(self.records)

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_cache"] = OrderedDict()
        return state

    def _open_image(self, relative_path: str, expected_size: tuple[int, int]):
        if relative_path in self._cache:
            image = self._cache.pop(relative_path)
            self._cache[relative_path] = image
            return image
        from PIL import Image

        file = _image_path(self.dataset_root, relative_path)
        with Image.open(file) as source:
            image = source.convert("RGB")
            image.load()
        if image.size != expected_size:
            raise ValueError(f"Image dimensions differ from manifest: {relative_path}")
        if self.cache_size:
            self._cache[relative_path] = image
            while len(self._cache) > self.cache_size:
                self._cache.popitem(last=False)
        return image

    def __getitem__(self, index: int):
        record = self.records[index]
        try:
            parent = self._open_image(
                record["relative_image_path"], (record["image_width"], record["image_height"])
            )
            crop = parent.crop(tuple(record[name] for name in ("crop_left", "crop_top", "crop_right", "crop_bottom")))
            if self.transform is not None:
                crop = self.transform(crop)
            return crop, int(record["class_index"])
        except (OSError, ValueError) as error:
            raise RuntimeError(f"Cannot load fruit {record['crop_id']} from {record['relative_image_path']}: {error}") from error
