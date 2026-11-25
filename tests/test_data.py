"""Data integrity checks run without torch, torchvision, or training downloads."""
import csv
import tempfile
import unittest
from pathlib import Path

from cafe_ml.data import CLASS_TO_INDEX, crop_bounds, load_records, prepare_crop_manifests


class FruitDataTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        (self.root / "csv").mkdir()
        self.images = [dict(split=s, image_id="1", relative_path=f"{s}/image.jpg",
            group_id=f"group_{s}", sha256=f"hash_{s}", source_name=f"source_{s}",
            width="640", height="640", annotation_count="1", capture_key=f"capture_{s}")
            for s in ("train", "valid", "test")]
        self.annotations = [dict(split=s, annotation_id="1", image_id="1", relative_image_path=f"{s}/image.jpg",
            group_id=f"group_{s}", category_id="4", label="semi_ripe", bbox_x="620.2", bbox_y="630.4",
            bbox_width="25.7", bbox_height="20.6", source_split=s, source_image_id="9", source_annotation_id="12")
            for s in ("train", "valid", "test")]
        self.save()

    def tearDown(self):
        self.temp.cleanup()

    def save(self):
        for name, rows in (("imagenes", self.images), ("anotaciones", self.annotations)):
            with (self.root / "csv" / f"{name}.csv").open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)

    def test_test_partition_requires_explicit_access(self):
        with self.assertRaisesRegex(ValueError, "reserved"):
            load_records(self.root, "test")
        self.assertEqual(len(load_records(self.root, "test", allow_test=True)), 1)

    def test_crop_bounds_cover_fractional_pixels_and_clip_edges(self):
        self.assertEqual(crop_bounds((-1.2, -3.1, 10.4, 12.3), 640, 640), (0, 0, 10, 10))
        self.assertEqual(crop_bounds((620.2, 630.4, 25.7, 20.6), 640, 640), (620, 630, 640, 640))

    def test_outside_nonfinite_and_degenerate_boxes_rejected(self):
        for box in ((700, 0, 3, 3), (0, 0, 0, 2), (float("nan"), 0, 3, 3), (-100, 0, 1, 2)):
            with self.subTest(box=box), self.assertRaises(ValueError):
                crop_bounds(box, 640, 640)

    def test_parent_group_leakage_is_rejected(self):
        self.images[1]["group_id"] = self.images[0]["group_id"]
        self.save()
        with self.assertRaisesRegex(ValueError, "Data leakage"):
            load_records(self.root, "train")

    def test_hash_leakage_is_rejected(self):
        self.images[1]["sha256"] = self.images[0]["sha256"]
        self.save()
        with self.assertRaisesRegex(ValueError, "Data leakage"):
            load_records(self.root, "train")

    def test_annotation_cannot_change_parent_split_or_group(self):
        self.annotations[0]["group_id"] = "group_test"
        self.save()
        with self.assertRaisesRegex(ValueError, "different group"):
            load_records(self.root, "train")

    def test_paths_cannot_escape_dataset(self):
        self.images[0]["relative_path"] = "../escape.jpg"
        self.save()
        with self.assertRaisesRegex(ValueError, "escapes"):
            load_records(self.root, "train")

    def test_fixed_labels_and_traceable_manifest_without_opening_images(self):
        records = load_records(self.root, "train")
        self.assertEqual(records[0]["class_index"], CLASS_TO_INDEX["semi_ripe"])
        self.assertEqual(records[0]["crop_id"], "train:1:1")
        self.assertEqual(records[0]["crop_right"], 640)
        summary = prepare_crop_manifests(self.root)
        self.assertEqual(set(summary), {"train", "valid"})
        self.assertEqual(summary["train"]["clipped_boxes"], 1)
        self.assertTrue((self.root / "csv" / "recortes_train.csv").exists())
        self.assertFalse((self.root / "csv" / "recortes_test.csv").exists())


if __name__ == "__main__":
    unittest.main()
