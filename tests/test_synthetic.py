"""Tests for the synthetic ground-truth generator."""

import json

import numpy as np
import pytest

from contour_shapes.errors import InvalidShapeError
from contour_shapes.synthetic import (
    SHAPE_NAMES,
    add_gaussian_noise,
    make_dataset,
    make_scene_image,
    make_shape_image,
)


def test_all_shape_names_render():
    for kind in SHAPE_NAMES:
        img, truth = make_shape_image(kind, seed=5)
        assert img.size == (256, 256)
        assert truth["label"] == kind
        assert (np.asarray(img) < 128).any()
        assert truth["bbox"][0] < truth["bbox"][2]
        assert truth["bbox"][1] < truth["bbox"][3]


def test_generation_is_deterministic():
    img1, t1 = make_shape_image("pentagon", seed=42)
    img2, t2 = make_shape_image("pentagon", seed=42)
    assert np.array_equal(np.asarray(img1), np.asarray(img2))
    assert t1 == t2


def test_different_seeds_differ():
    img1, _ = make_shape_image("circle", seed=1)
    img2, _ = make_shape_image("circle", seed=2)
    assert not np.array_equal(np.asarray(img1), np.asarray(img2))


def test_invalid_shape_raises():
    with pytest.raises(InvalidShapeError):
        make_shape_image("hexagon", seed=0)


def test_rotation_changes_pixels():
    img1, _ = make_shape_image("square", seed=5, rotation_deg=0.0)
    img2, _ = make_shape_image("square", seed=5, rotation_deg=45.0)
    assert not np.array_equal(np.asarray(img1), np.asarray(img2))


def test_noise_changes_pixels_but_keeps_shape():
    clean, _ = make_shape_image("circle", seed=5)
    noisy, _ = make_shape_image("circle", seed=5, noise_sigma=25.0)
    assert not np.array_equal(np.asarray(clean), np.asarray(noisy))
    # Otsu-level check: both masks are almost the same area.
    assert abs((np.asarray(clean) < 128).sum() - (np.asarray(noisy) < 128).sum()) < 500


def test_add_gaussian_noise_zero_is_identity():
    arr = np.full((10, 10), 128, dtype=np.uint8)
    rng = np.random.default_rng(0)
    assert np.array_equal(add_gaussian_noise(arr, 0.0, rng), arr)


def test_scene_contains_all_shapes():
    kinds = ["triangle", "square", "circle"]
    img, truths = make_scene_image(kinds, size=384, seed=7)
    assert len(truths) == 3
    assert [t["label"] for t in truths] == kinds
    assert (np.asarray(img) < 128).sum() > 1000


def test_make_dataset_writes_ground_truth(tmp_path):
    gt_path = make_dataset(tmp_path, count=12, seed=0)
    records = json.loads(gt_path.read_text(encoding="utf-8"))
    assert len(records) == 12
    assert [r["label"] for r in records] == [SHAPE_NAMES[i % 6] for i in range(12)]
    for rec in records:
        assert (tmp_path / rec["file"]).is_file()
