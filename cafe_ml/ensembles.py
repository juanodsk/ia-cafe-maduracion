"""Soft-voting ensembles of aligned per-fruit prediction files.

Weights must be fixed in advance or selected on validation. This module does
not optimize weights or choose models using test results.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .evaluation import LABELS, _identity, _validated_inputs, save_evaluation


def ensemble_predictions(
    prediction_paths: Sequence[str | Path],
    output_dir: str | Path,
    *,
    weights: Sequence[float] | None = None,
    labels: Sequence[str] = LABELS,
    model_name: str = "soft_voting",
    allow_test: bool = False,
    save_plot: bool = True,
) -> dict[str, Any]:
    """Average probabilities after strict checks of ordered fruit identities.

    All files must contain the same rows, in the same order, with matching
    truth labels, split and parent groups. Rejecting reordered rows makes a
    potentially incorrect model-output alignment explicit.
    """
    paths = [Path(value) for value in prediction_paths]
    if len(paths) < 2:
        raise ValueError("An ensemble requires prediction files from at least two models.")
    if len({path.resolve() for path in paths}) != len(paths):
        raise ValueError("An ensemble must not list the same prediction file twice.")
    weights = np.ones(len(paths), dtype=np.float64) if weights is None else np.asarray(weights, dtype=np.float64)
    if weights.shape != (len(paths),) or not np.isfinite(weights).all() or (weights < 0).any() or weights.sum() <= 0:
        raise ValueError("Weights must be finite, nonnegative, match the models and have a positive sum.")
    weights = weights / weights.sum()
    reference = None
    reference_keys = None
    reference_truth = None
    averaged = None
    for model_index, path in enumerate(paths):
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            required = {"split", "image_id", "annotation_id", "group_id", "true_label"}
            required.update("prob_" + label for label in labels)
            if not required.issubset(reader.fieldnames or []):
                raise ValueError(f"Prediction CSV {path} is missing required columns.")
            rows = list(reader)
        if not rows:
            raise ValueError(f"Prediction CSV is empty: {path}.")
        if any(row["split"] == "test" for row in rows) and not allow_test:
            raise ValueError("Test ensembles are reserved for final evaluation; pass allow_test=True explicitly.")
        keys = [_identity(row) for row in rows]
        if len(set(keys)) != len(keys):
            raise ValueError(f"Duplicate fruit identities in {path}.")
        truth = [row["true_label"] for row in rows]
        probabilities = [[float(row["prob_" + label]) for label in labels] for row in rows]
        _, probabilities, _ = _validated_inputs(truth, probabilities, labels)
        if reference is None:
            reference = rows
            reference_keys = keys
            reference_truth = truth
            averaged = np.zeros_like(probabilities)
        elif keys != reference_keys:
            raise ValueError(f"Fruit identity/order mismatch in {path}; align predictions before ensembling.")
        elif truth != reference_truth:
            raise ValueError(f"Ground-truth labels disagree between model predictions: {path}.")
        averaged += weights[model_index] * probabilities
    records = [{**row, "label": row["true_label"]} for row in reference]
    return save_evaluation(
        output_dir, records, averaged, labels=labels, model_name=model_name,
        allow_test=allow_test, save_plot=save_plot,
        metadata={
            "method": "soft_voting",
            "prediction_files": [str(path) for path in paths],
            "normalized_weights": weights.tolist(),
            "weight_selection": "caller_provided_or_uniform; no_test_optimization",
        },
    )
