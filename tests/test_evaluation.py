"""Small deterministic checks of metric definitions and ensemble row alignment."""

import csv
import tempfile
from pathlib import Path
import unittest

import numpy as np

from cafe_ml.evaluation import LABELS, evaluate_predictions, save_evaluation
from cafe_ml.ensembles import ensemble_predictions


class EvaluationTests(unittest.TestCase):
    def probabilities(self):
        # Five targets, one of each class; the second is confused with dry.
        return np.asarray([
            [0.8, 0.05, 0.05, 0.05, 0.05],
            [0.6, 0.2, 0.05, 0.05, 0.1],
            [0.05, 0.05, 0.8, 0.05, 0.05],
            [0.05, 0.05, 0.05, 0.8, 0.05],
            [0.05, 0.05, 0.05, 0.05, 0.8],
        ])

    def records(self, split="valid"):
        return [
            {"split": split, "image_id": i + 10, "annotation_id": i + 100,
             "group_id": "group_" + str(i), "label": label}
            for i, label in enumerate(LABELS)
        ]

    def test_confusion_and_macro_f1_include_all_classes(self):
        metrics = evaluate_predictions(list(LABELS), self.probabilities())
        self.assertEqual(metrics["confusion_matrix"], [
            [1, 0, 0, 0, 0], [1, 0, 0, 0, 0], [0, 0, 1, 0, 0],
            [0, 0, 0, 1, 0], [0, 0, 0, 0, 1],
        ])
        self.assertAlmostEqual(metrics["accuracy"], 0.8)
        self.assertAlmostEqual(metrics["balanced_accuracy"], 0.8)
        self.assertAlmostEqual(metrics["macro_f1"], ((2 / 3) + 0 + 1 + 1 + 1) / 5)
        self.assertEqual(sum(row["support"] for row in metrics["per_class"]), 5)

    def test_probability_inputs_reject_logits_and_nonfinite_values(self):
        with self.assertRaisesRegex(ValueError, "sum to 1"):
            evaluate_predictions([0], [[0.1] * 5])
        with self.assertRaisesRegex(ValueError, "NaN"):
            evaluate_predictions([0], [[np.nan, 0, 0, 0, 1]])
        with self.assertRaisesRegex(ValueError, "row counts"):
            evaluate_predictions([0, 1], [[1, 0, 0, 0, 0]])

    def test_test_split_requires_explicit_final_evaluation(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaisesRegex(ValueError, "reserved for final"):
                save_evaluation(folder, self.records("test"), self.probabilities(), save_plot=False)
            self.assertFalse((Path(folder) / "predictions.csv").exists())

    def test_csv_keeps_fruit_identity_and_probability_label_order(self):
        with tempfile.TemporaryDirectory() as folder:
            save_evaluation(folder, self.records(), self.probabilities(), save_plot=False)
            with (Path(folder) / "predictions.csv").open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(rows[1]["annotation_id"], "101")
            self.assertEqual(rows[1]["group_id"], "group_1")
            self.assertEqual(rows[1]["true_label"], "overripe")
            self.assertEqual(rows[1]["pred_label"], "dry")
            self.assertAlmostEqual(float(rows[1]["prob_overripe"]), 0.2)
            self.assertTrue((Path(folder) / "confusion_matrix.csv").is_file())

    def test_soft_voting_validates_identity_order(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            save_evaluation(root / "first", self.records(), self.probabilities(), save_plot=False)
            save_evaluation(root / "second", self.records(), self.probabilities(), save_plot=False)
            metrics = ensemble_predictions(
                [root / "first/predictions.csv", root / "second/predictions.csv"], root / "ensemble", save_plot=False,
            )
            self.assertAlmostEqual(metrics["accuracy"], 0.8)
            reordered_records = list(reversed(self.records()))
            save_evaluation(root / "reordered", reordered_records, self.probabilities()[::-1], save_plot=False)
            with self.assertRaisesRegex(ValueError, "identity/order mismatch"):
                ensemble_predictions(
                    [root / "first/predictions.csv", root / "reordered/predictions.csv"], root / "bad", save_plot=False,
                )

    def test_soft_voting_rejects_truth_disagreement_and_invalid_weights(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            save_evaluation(root / "first", self.records(), self.probabilities(), save_plot=False)
            altered = self.records()
            altered[0]["label"] = "ripe"
            save_evaluation(root / "altered", altered, self.probabilities(), save_plot=False)
            paths = [root / "first/predictions.csv", root / "altered/predictions.csv"]
            with self.assertRaisesRegex(ValueError, "Ground-truth labels disagree"):
                ensemble_predictions(paths, root / "bad_truth", save_plot=False)
            with self.assertRaisesRegex(ValueError, "Weights must"):
                ensemble_predictions(paths, root / "bad_weights", weights=[-1, 2], save_plot=False)


if __name__ == "__main__":
    unittest.main()
