"""Evaluation of classification results against ground truth.

Produces a confusion matrix plus per-class precision/recall/F1 and
overall accuracy — the same numbers the CLI ``evaluate`` command prints
and the ``examples/RESULTS.md`` report is built from.  All metrics are
computed from the raw ``(true_label, predicted_label)`` pairs; nothing
is smoothed or rounded before the division.
"""

from __future__ import annotations

from .classify import CLASSES

__all__ = ["confusion_matrix", "classification_report", "format_report"]


def confusion_matrix(
    pairs: list[tuple[str, str]], labels: list[str] | None = None
) -> tuple[list[str], list[list[int]]]:
    """Confusion matrix for ``(true, predicted)`` label pairs.

    Returns ``(labels, matrix)`` where ``matrix[i][j]`` counts pairs
    with true label ``labels[i]`` predicted as ``labels[j]``.  Rows are
    true labels, columns are predictions.  Labels outside ``labels``
    (e.g. ``"unknown"``) get their own extra row/column.
    """
    labels = list(labels) if labels is not None else list(CLASSES)
    extra = sorted({p for pair in pairs for p in pair} - set(labels))
    all_labels = labels + extra
    index = {name: i for i, name in enumerate(all_labels)}
    matrix = [[0] * len(all_labels) for _ in all_labels]
    for true, pred in pairs:
        matrix[index[true]][index[pred]] += 1
    return all_labels, matrix


def classification_report(pairs: list[tuple[str, str]]) -> dict:
    """Accuracy plus per-class precision/recall/F1 for label pairs.

    Returns a dict with ``accuracy``, ``total``, ``correct`` and a
    ``per_class`` mapping ``label -> {precision, recall, f1, support}``.
    A class with no predictions gets precision 0.0; a class with no
    true samples gets recall 0.0 (both documented, not hidden).
    """
    labels, matrix = confusion_matrix(pairs)
    total = len(pairs)
    correct = sum(matrix[i][i] for i in range(len(labels)))
    per_class: dict[str, dict[str, float | int]] = {}
    for i, name in enumerate(labels):
        tp = matrix[i][i]
        predicted = sum(matrix[r][i] for r in range(len(labels)))
        actual = sum(matrix[i])
        precision = tp / predicted if predicted else 0.0
        recall = tp / actual if actual else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        per_class[name] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "support": actual,
        }
    return {
        "accuracy": round(correct / total, 4) if total else 0.0,
        "total": total,
        "correct": correct,
        "per_class": per_class,
    }


def format_report(pairs: list[tuple[str, str]]) -> str:
    """Pretty text report: confusion matrix + per-class metrics."""
    labels, matrix = confusion_matrix(pairs)
    report = classification_report(pairs)
    width = max(len(name) for name in labels)
    lines = ["Confusion matrix (rows = true, columns = predicted):"]
    header = " " * (width + 2) + " ".join(f"{name:>9}" for name in labels)
    lines.append(header)
    for i, name in enumerate(labels):
        row = " ".join(f"{matrix[i][j]:>9}" for j in range(len(labels)))
        lines.append(f"{name:>{width}}  {row}")
    lines.append("")
    lines.append(f"Accuracy: {report['correct']}/{report['total']} = {report['accuracy']:.4f}")
    lines.append("")
    lines.append(f"{'class':<{width}}  {'precision':>9}  {'recall':>9}  {'f1':>9}  {'support':>7}")
    for name in labels:
        m = report["per_class"][name]
        lines.append(
            f"{name:<{width}}  {m['precision']:>9.4f}  {m['recall']:>9.4f}  {m['f1']:>9.4f}  {m['support']:>7}"
        )
    return "\n".join(lines)
