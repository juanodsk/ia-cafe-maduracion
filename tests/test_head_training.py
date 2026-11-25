"""CPU checks for train-only statistics, saved heads and grouped CV."""

import csv
import tempfile
from pathlib import Path
import unittest

import numpy as np
import optuna
import torch

from cafe_ml.data import CLASS_NAMES
from cafe_ml.head_training import (
    compute_class_weights,
    cross_validate_head,
    load_head_checkpoint,
    predict_head,
    run_head_study,
    save_head_checkpoint,
    train_head,
    tune_head,
)


def fixture(split="train", groups=10, seed=42):
    rng = np.random.default_rng(seed)
    records, features = [], []
    for group in range(groups):
        for label, label_name in enumerate(CLASS_NAMES):
            row = len(records)
            records.append({"split": split, "image_id": group + 1, "annotation_id": row + 1,
                            "group_id": f"{split}-g{group}", "label": label_name,
                            "class_index": label, "crop_id": f"{split}:{group + 1}:{row + 1}"})
            features.append(np.eye(5)[label] + rng.normal(0, 0.03, 5))
    return np.asarray(features, dtype=np.float16), records


class HeadTrainingTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.params = {"lr": 0.01, "batch_size": 16, "weight_decay": 1e-4,
                       "dropout": 0.2, "hidden_dim": 0, "class_weight_power": 0.5}

    def test_statistics_use_training_only_and_checkpoint_roundtrip(self):
        training, records = fixture(groups=4)
        validation, valid_records = fixture("valid", groups=2)
        validation = validation + 50
        labels = np.array([r["class_index"] for r in records])
        valid_labels = np.array([r["class_index"] for r in valid_records])
        head, history, metrics = train_head(training, labels, validation, valid_labels,
                                            self.params, device="cpu", max_epochs=2, seed=17)
        np.testing.assert_allclose(head.feature_mean.detach().numpy(), training.astype(np.float32).mean(axis=0), atol=1e-6)
        self.assertEqual(len(history), 2)
        self.assertIn(metrics["epoch"], (1, 2))
        expected = predict_head(head, validation)
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "head.pt"
            save_head_checkpoint(checkpoint, head, {"normalization_fit": "train_only"})
            loaded = load_head_checkpoint(checkpoint)
            np.testing.assert_array_equal(predict_head(loaded, validation), expected)
            self.assertEqual(loaded.checkpoint_metadata["normalization_fit"], "train_only")

    def test_class_weights_follow_only_supplied_subset(self):
        labels = np.repeat(np.arange(5), [20, 10, 8, 6, 4])
        actual = compute_class_weights(labels, power=0.5)
        expected = np.asarray([20, 10, 8, 6, 4], dtype=np.float64) ** -0.5
        expected /= expected.mean()
        np.testing.assert_allclose(actual, expected, rtol=1e-6)
        with self.assertRaises(ValueError):
            compute_class_weights(np.array([0, 1, 2, 3]), power=1)

    def test_cv_groups_are_exclusive_and_normalization_is_per_fold(self):
        features, records = fixture(groups=6)
        with tempfile.TemporaryDirectory() as directory:
            summary = cross_validate_head(features, records, self.params, directory,
                                           device="cpu", cv_folds=3, epochs=2)
            self.assertEqual(summary["fixed_epochs"], 2)
            self.assertEqual(len(summary["cv_checkpoint_paths"]), 3)
            with (Path(directory) / "fold_assignments.csv").open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), len(records))
            group_folds = {}
            for row in rows:
                self.assertEqual(row["split"], "train")
                group_folds.setdefault(row["group_id"], set()).add(int(row["fold"]))
            self.assertTrue(all(len(folds) == 1 for folds in group_folds.values()))
            for fold, checkpoint in enumerate(summary["cv_checkpoint_paths"], start=1):
                fit_rows = [int(row["row_index"]) for row in rows if int(row["fold"]) != fold]
                head = load_head_checkpoint(checkpoint)
                expected_mean = features[fit_rows].astype(np.float32).mean(axis=0)
                np.testing.assert_allclose(head.feature_mean.numpy(), expected_mean, atol=1e-6)
                self.assertEqual(head.checkpoint_metadata["epochs"], 2)

    def test_test_records_are_rejected_before_tuning(self):
        features, records = fixture(groups=2)
        held_features, held_records = fixture("test", groups=2)
        with tempfile.TemporaryDirectory() as directory, self.assertRaises(ValueError):
            tune_head(features, records, held_features, held_records, directory, n_trials=1, max_epochs=1)

    def test_small_hyperband_study_produces_reproducible_selected_head(self):
        features, records = fixture(groups=4)
        valid_features, valid_records = fixture("valid", groups=2, seed=99)
        with tempfile.TemporaryDirectory() as directory:
            result = run_head_study(features, records, valid_features, valid_records,
                                    directory, device="cpu", seed=11, n_trials=2,
                                    max_epochs=2, cv_folds=2, model_name="offline_test")
            self.assertFalse(result["test_accessed"])
            self.assertTrue(Path(result["checkpoint_path"]).exists())
            self.assertEqual(len(result["cv_checkpoint_paths"]), 2)
            self.assertTrue((Path(directory) / "hyperband" / "study.sqlite3").exists())
            self.assertTrue((Path(directory) / "hyperband" / "study_trials.csv").exists())
            loaded = load_head_checkpoint(result["checkpoint_path"])
            probabilities = predict_head(loaded, valid_features)
            self.assertEqual(probabilities.shape, (len(valid_records), 5))
            np.testing.assert_allclose(probabilities.sum(axis=1), 1, atol=1e-6)

    def test_resumed_random_sampler_matches_uninterrupted_study(self):
        features, records = fixture(groups=3)
        valid_features, valid_records = fixture("valid", groups=2, seed=99)
        with tempfile.TemporaryDirectory() as directory:
            resumed = Path(directory) / "resumed"
            fresh = Path(directory) / "fresh"
            for folder, count in ((resumed, 1), (resumed, 3), (fresh, 3)):
                tune_head(features, records, valid_features, valid_records, folder,
                          device="cpu", seed=19, n_trials=count, max_epochs=1,
                          model_name="resume_test")
            params = []
            for folder in (resumed, fresh):
                with (folder / "study_trials.csv").open(newline="", encoding="utf-8") as handle:
                    rows = list(csv.DictReader(handle))
                params.append([{key: value for key, value in row.items() if key != "state"} for row in rows])
            self.assertEqual(params[0], params[1])


if __name__ == "__main__":
    unittest.main()
