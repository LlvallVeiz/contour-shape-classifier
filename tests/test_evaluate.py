"""Tests for confusion matrix and classification report."""

from contour_shapes.evaluate import classification_report, confusion_matrix, format_report


def test_confusion_matrix_counts():
    pairs = [("circle", "circle"), ("circle", "square"), ("square", "square")]
    labels, matrix = confusion_matrix(pairs, labels=["circle", "square"])
    assert labels == ["circle", "square"]
    assert matrix == [[1, 1], [0, 1]]


def test_confusion_matrix_unknown_gets_extra_column():
    pairs = [("circle", "unknown"), ("circle", "circle")]
    labels, matrix = confusion_matrix(pairs)
    assert "unknown" in labels
    assert matrix[labels.index("circle")][labels.index("unknown")] == 1


def test_classification_report_perfect():
    pairs = [("a", "a"), ("a", "a"), ("b", "b")]
    report = classification_report(pairs)
    assert report["accuracy"] == 1.0
    assert report["correct"] == 3
    assert report["total"] == 3
    assert report["per_class"]["a"]["precision"] == 1.0
    assert report["per_class"]["a"]["recall"] == 1.0


def test_classification_report_partial():
    pairs = [("a", "a"), ("a", "b"), ("b", "b"), ("b", "a")]
    report = classification_report(pairs)
    assert report["accuracy"] == 0.5
    assert report["per_class"]["a"]["precision"] == 0.5
    assert report["per_class"]["a"]["recall"] == 0.5
    assert report["per_class"]["a"]["f1"] == 0.5


def test_format_report_contains_accuracy():
    pairs = [("circle", "circle"), ("square", "circle")]
    text = format_report(pairs)
    assert "Accuracy: 1/2 = 0.5000" in text
    assert "circle" in text and "square" in text
