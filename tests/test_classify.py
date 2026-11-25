"""Tests for the rule-based classifier."""

import numpy as np
import pytest

from contour_shapes.classify import CLASSES, analyze_mask, class_templates, classify_features
from contour_shapes.features import extract_features
from contour_shapes.contours import find_contours
from contour_shapes.synthetic import make_scene_image, make_shape_image


def _analyse(kind: str, **kwargs):
    img, _truth = make_shape_image(kind, **kwargs)
    mask = np.asarray(img) < 128
    results = analyze_mask(mask)
    assert len(results) == 1
    return results[0]


@pytest.mark.parametrize("kind", CLASSES)
def test_clean_shape_classified_correctly(kind):
    res = _analyse(kind, seed=11)
    assert res.label == kind
    assert res.confidence >= 0.7


@pytest.mark.parametrize("kind", CLASSES)
def test_rotated_shape_classified_correctly(kind):
    res = _analyse(kind, seed=21, rotation_deg=137.0)
    assert res.label == kind
    assert res.confidence >= 0.6


@pytest.mark.parametrize("kind", CLASSES)
def test_noisy_shape_classified_correctly(kind):
    res = _analyse(kind, seed=31, noise_sigma=20.0)
    assert res.label == kind
    assert res.confidence >= 0.55


def test_small_shape_classified_correctly():
    res = _analyse("circle", seed=41, size=128)
    assert res.label == "circle"


def test_templates_cover_all_classes():
    templates = class_templates()
    assert set(templates) == set(CLASSES)
    for kind, tpl in templates.items():
        assert 0.4 < tpl["circularity"] < 1.0
        assert tpl["elongation"] > 0.9
        assert np.all(np.isfinite(tpl["hu_log"]))


def test_templates_are_stable_across_calls():
    first = class_templates()["circle"]["circularity"]
    second = class_templates()["circle"]["circularity"]
    assert first == second  # lazy cache, not recomputed


def test_scores_are_probability_like():
    res = _analyse("square", seed=11)
    assert all(0.0 <= v <= 1.0 for v in res.scores.values())
    assert set(res.scores) == set(CLASSES)
    assert res.scores["square"] == pytest.approx(res.confidence)


def test_tiny_square_blob_is_a_square():
    # A 3x3 filled blob really is a (tiny) square: 4 vertices,
    # circularity ~0.785, elongation 1.0.  Classifying it as a square
    # is correct behaviour, not a failure of the "unknown" path.
    mask = np.zeros((64, 64), dtype=bool)
    mask[30:33, 30:33] = True  # 3x3 blob, 9 px
    results = analyze_mask(mask, min_area=1)
    assert len(results) == 1
    assert results[0].label == "square"


def test_concave_shapes_are_unknown():
    # L- and plus-shaped blobs are concave, so the solidity gate
    # (all six classes are convex) must report them as unknown instead
    # of forcing them into the closest convex class.
    l_mask = np.zeros((64, 64), dtype=bool)
    l_mask[20:40, 20:25] = True
    l_mask[35:40, 20:45] = True
    assert analyze_mask(l_mask, min_area=1)[0].label == "unknown"
    plus_mask = np.zeros((64, 64), dtype=bool)
    plus_mask[25:40, 30:35] = True
    plus_mask[30:35, 25:40] = True
    assert analyze_mask(plus_mask, min_area=1)[0].label == "unknown"


def test_scene_finds_all_shapes():
    img, truths = make_scene_image(
        ["triangle", "square", "rectangle", "pentagon", "circle", "ellipse"],
        size=512,
        seed=99,
    )
    results = analyze_mask(np.asarray(img) < 128)
    assert len(results) == 6
    assert sorted(r.label for r in results) == sorted(t["label"] for t in truths)


def test_empty_mask_gives_no_results():
    assert analyze_mask(np.zeros((50, 50), dtype=bool)) == []
