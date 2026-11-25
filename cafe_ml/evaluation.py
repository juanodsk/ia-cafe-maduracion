"""Classification metrics and traceable, per-fruit evaluation artifacts.

The training runner must select checkpoints using validation only. Access to
test predictions is explicit through ``allow_test=True`` at the final stage.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Mapping, Sequence
import warnings

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)


LABELS = ("dry", "overripe", "ripe", "semi_ripe", "unripe")
IDENTITY_FIELDS = ("split", "image_id", "annotation_id", "group_id")
TRACE_FIELDS = (
    "relative_image_path",
    "category_id",
    "source_split",
    "source_image_id",
    "source_annotation_id",
)


def _check_labels(labels: Sequence[str]) -> tuple[str, ...]:
    labels = tuple(labels)
    if not labels or any(not isinstance(label, str) or not label for label in labels):
        raise ValueError("Class labels must be nonempty strings.")
    if len(set(labels)) != len(labels):
        raise ValueError("Class labels must be unique and in a fixed order.")
    return labels


def _validated_inputs(
    y_true: Sequence[int | str],
    probabilities: Any,
    labels: Sequence[str],
) -> tuple[np.ndarray, np.ndarray, tuple[str, ...]]:
    labels = _check_labels(labels)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if probabilities.ndim != 2 or probabilities.shape[1] != len(labels):
        raise ValueError(f"Expected probability shape (n, {len(labels)}).")
    if probabilities.shape[0] == 0:
        raise ValueError("Cannot evaluate an empty dataset.")
    if not np.isfinite(probabilities).all():
        raise ValueError("Predicted probabilities contain NaN or infinity.")
    if (probabilities < 0).any() or (probabilities > 1).any():
        raise ValueError("Predicted probabilities must be between 0 and 1.")
    if not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-4, rtol=0):
        raise ValueError("Each probability row must sum to 1; pass softmax outputs, not logits.")
    targets = []
    for target in y_true:
        if isinstance(target, str):
            if target not in labels:
                raise ValueError(f"Unknown ground-truth label: {target!r}.")
            targets.append(labels.index(target))
        elif isinstance(target, (int, np.integer)) and not isinstance(target, (bool, np.bool_)):
            if not 0 <= int(target) < len(labels):
                raise ValueError(f"Ground-truth class index out of range: {target}.")
            targets.append(int(target))
        else:
            raise ValueError("Ground truth must contain class names or integer class indices.")
    targets = np.asarray(targets, dtype=np.int64)
    if targets.shape[0] != probabilities.shape[0]:
        raise ValueError("Ground-truth and probability row counts differ.")
    return targets, probabilities, labels


def evaluate_predictions(
    y_true: Sequence[int | str],
    probabilities: Any,
    labels: Sequence[str] = LABELS,
) -> dict[str, Any]:
    """Return sklearn metrics, with macro F1 over the complete class vocabulary.

    Balanced accuracy averages recall over the classes present in ground truth,
    following sklearn's definition. Absent classes still appear in the report
    and confusion matrix, and contribute zero to macro F1.
    """
    targets, probabilities, labels = _validated_inputs(y_true, probabilities, labels)
    predicted = probabilities.argmax(axis=1)
    indices = np.arange(len(labels))
    precision, recall, f1, support = precision_recall_fscore_support(
        targets, predicted, labels=indices, average=None, zero_division=0
    )
    _, _, weighted_f1, _ = precision_recall_fscore_support(
        targets, predicted, labels=indices, average="weighted", zero_division=0
    )
    # sklearn warns when a predicted class is absent from a small validation
    # subset. Its documented present-class definition remains appropriate.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="y_pred contains classes not in y_true", category=UserWarning)
        balanced = balanced_accuracy_score(targets, predicted)
    return {
        "num_samples": int(targets.size),
        "class_labels": list(labels),
        "accuracy": float(accuracy_score(targets, predicted)),
        "balanced_accuracy": float(balanced),
        "macro_f1": float(f1.mean()),
        "weighted_f1": float(weighted_f1),
        "macro_precision": float(precision.mean()),
        "macro_recall": float(recall.mean()),
        "confusion_matrix": confusion_matrix(targets, predicted, labels=indices).tolist(),
        "per_class": [
            {
                "label": label,
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1": float(f1[index]),
                "support": int(support[index]),
            }
            for index, label in enumerate(labels)
        ],
    }


def _identity(record: Mapping[str, Any]) -> tuple[str, ...]:
    values = tuple(str(record.get(field, "")) for field in IDENTITY_FIELDS)
    if any(value in ("", "None") for value in values):
        raise ValueError(f"Each prediction record requires {IDENTITY_FIELDS}.")
    return values


def _write_csv(path: Path, fields: Sequence[str], rows: Sequence[Mapping[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _save_confusion_plot(path: Path, matrix: Any, labels: Sequence[str], title: str) -> None:
    import matplotlib

    matplotlib.use("Agg", force=True)
    from matplotlib import pyplot as plt

    matrix = np.asarray(matrix)
    figure, axis = plt.subplots(figsize=(7.6, 6.5), layout="constrained")
    picture = axis.imshow(matrix, cmap="Blues", interpolation="nearest")
    figure.colorbar(picture, ax=axis, shrink=0.85, label="Frutos")
    axis.set(
        xticks=np.arange(len(labels)),
        yticks=np.arange(len(labels)),
        xticklabels=labels,
        yticklabels=labels,
        xlabel="Clase predicha",
        ylabel="Clase real",
        title=title,
    )
    plt.setp(axis.get_xticklabels(), rotation=35, ha="right", rotation_mode="anchor")
    threshold = matrix.max() / 2.0
    for row in range(len(labels)):
        for column in range(len(labels)):
            axis.text(
                column, row, str(int(matrix[row, column])), ha="center", va="center",
                color="white" if matrix[row, column] > threshold else "black",
            )
    figure.savefig(path, dpi=170)
    plt.close(figure)


def save_evaluation(
    output_dir: str | Path,
    records: Sequence[Mapping[str, Any]],
    probabilities: Any,
    *,
    labels: Sequence[str] = LABELS,
    model_name: str | None = None,
    allow_test: bool = False,
    save_plot: bool = True,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Save metrics and one probability row per annotated fruit.

    Record order must match the model probability order. Rows inherit the
    parent image's saved partition and group; this function never repartitions
    fruits. Test evaluation is rejected unless explicitly authorized by the
    final evaluation stage through ``allow_test=True``.
    """
    records = list(records)
    if not records:
        raise ValueError("Cannot save evaluation without prediction records.")
    identities = [_identity(record) for record in records]
    if len(set(identities)) != len(identities):
        raise ValueError("Duplicate fruit identities in prediction records.")
    splits = {identity[0] for identity in identities}
    if len(splits) != 1:
        raise ValueError("Save each evaluation partition separately.")
    split = next(iter(splits))
    if split == "test" and not allow_test:
        raise ValueError("Test is reserved for final evaluation; pass allow_test=True explicitly.")
    truth = [record.get("label", record.get("true_label")) for record in records]
    targets, probabilities, labels = _validated_inputs(truth, probabilities, labels)
    metrics = evaluate_predictions(targets, probabilities, labels)
    metrics["evaluation_split"] = split
    if model_name is not None:
        metrics["model_name"] = model_name
    if metadata is not None:
        metrics["metadata"] = dict(metadata)
    predicted = probabilities.argmax(axis=1)
    prediction_rows = []
    trace_fields = [field for field in TRACE_FIELDS if any(field in record for record in records)]
    for index, record in enumerate(records):
        row = {field: record[field] for field in IDENTITY_FIELDS}
        row.update({field: record.get(field, "") for field in trace_fields})
        row.update(
            true_label=labels[targets[index]], pred_label=labels[predicted[index]],
            true_index=int(targets[index]), pred_index=int(predicted[index]),
        )
        row.update({"prob_" + label: float(probabilities[index, c]) for c, label in enumerate(labels)})
        prediction_rows.append(row)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    fields = [*IDENTITY_FIELDS, *trace_fields, "true_label", "pred_label", "true_index", "pred_index"]
    fields.extend("prob_" + label for label in labels)
    _write_csv(output_dir / "predictions.csv", fields, prediction_rows)
    _write_csv(output_dir / "classification_report.csv", ("label", "precision", "recall", "f1", "support"), metrics["per_class"])
    _write_csv(
        output_dir / "confusion_matrix.csv", ["true_label", *labels],
        [{"true_label": label, **dict(zip(labels, metrics["confusion_matrix"][i]))} for i, label in enumerate(labels)],
    )
    if save_plot:
        title = "Matriz de confusión" + (" — " + model_name if model_name else "")
        _save_confusion_plot(output_dir / "confusion_matrix.png", metrics["confusion_matrix"], labels, title)
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    return metrics
