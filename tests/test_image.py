"""Tests for image loading and Otsu binarisation."""

import numpy as np
import pytest
from PIL import Image

from contour_shapes.errors import EmptyImageError, ImageLoadError
from contour_shapes.image import binarize, load_binary, load_grayscale, otsu_threshold


def test_otsu_gradient_is_midpoint():
    """A uniform gradient 0..255 splits at the middle."""
    gray = np.tile(np.arange(256, dtype=np.uint8), (64, 1))
    thr = otsu_threshold(gray)
    assert 120 <= thr <= 135


def test_otsu_bimodal_separates_modes():
    """Otsu on a two-mode image recovers the dark class exactly."""
    rng = np.random.default_rng(0)
    dark = rng.integers(40, 61, size=(50, 100)).astype(np.uint8)
    bright = rng.integers(190, 211, size=(50, 100)).astype(np.uint8)
    gray = np.vstack([dark, bright])
    thr = otsu_threshold(gray)
    assert 60 <= thr < 190  # any threshold between the modes separates them
    mask = binarize(gray, threshold=thr)
    assert mask[:50].all() and not mask[50:].any()


def test_binarize_invert_flips_foreground():
    gray = np.zeros((10, 10), dtype=np.uint8)
    gray[:, 5:] = 255
    assert binarize(gray, threshold=127).sum() == 50
    assert binarize(gray, threshold=127, invert=True).sum() == 50
    assert binarize(gray, threshold=127, invert=True)[0, 0] == False
    assert binarize(gray, threshold=127, invert=True)[0, 9] == True


def test_load_grayscale_missing_file(tmp_path):
    with pytest.raises(ImageLoadError):
        load_grayscale(tmp_path / "nope.png")


def test_load_grayscale_not_an_image(tmp_path):
    bad = tmp_path / "bad.png"
    bad.write_text("this is not an image", encoding="utf-8")
    with pytest.raises(ImageLoadError):
        load_grayscale(bad)


def test_load_binary_roundtrip(tmp_path):
    img = Image.new("L", (40, 40), 255)
    px = img.load()
    for y in range(10, 30):
        for x in range(10, 30):
            px[x, y] = 0
    path = tmp_path / "square.png"
    img.save(path)
    mask, thr = load_binary(path)
    assert mask.shape == (40, 40)
    assert mask.sum() == 400
    assert 0 <= thr <= 255


def test_load_binary_empty_image(tmp_path):
    path = tmp_path / "white.png"
    Image.new("L", (20, 20), 255).save(path)
    with pytest.raises(EmptyImageError):
        load_binary(path)
