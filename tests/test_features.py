"""Tests for geometric features and Hu moments."""

import numpy as np
import pytest
from PIL import Image

from contour_shapes.features import (
    approximate_polygon,
    circularity,
    convex_hull,
    extract_features,
    hu_moments,
    log_hu_moments,
    polygon_area,
    polygon_perimeter,
)
from contour_shapes.contours import find_contours
from contour_shapes.synthetic import make_shape_image


def _square_points(side=100):
    return np.array([[0, 0], [side, 0], [side, side], [0, side], [0, 0]])


def test_shoelace_area_square():
    assert polygon_area(_square_points(100)) == pytest.approx(10000.0)


def test_shoelace_area_triangle():
    tri = np.array([[0, 0], [10, 0], [0, 10], [0, 0]])
    assert polygon_area(tri) == pytest.approx(50.0)


def test_perimeter_square():
    assert polygon_perimeter(_square_points(100)) == pytest.approx(400.0)


def test_circularity_circle_is_high():
    yy, xx = np.mgrid[0:101, 0:101]
    mask = (xx - 50) ** 2 + (yy - 50) ** 2 <= 30**2
    pts = find_contours(mask)[0].points
    circ = circularity(polygon_area(pts), polygon_perimeter(pts))
    assert 0.85 <= circ <= 1.0


def test_circularity_square_is_pi_over_4():
    pts = _square_points(100)
    assert circularity(polygon_area(pts), polygon_perimeter(pts)) == pytest.approx(
        np.pi / 4, abs=0.01
    )


def test_rdp_straight_line_collapses():
    line = np.column_stack([np.arange(50), np.zeros(50, dtype=int)])
    assert len(approximate_polygon(line, epsilon=1.0)) == 2


def test_rdp_square_keeps_four_corners():
    pts = _square_points(100)
    # Densify the edges so the trace looks like a real contour.
    dense = []
    for a, b in zip(pts[:-1], pts[1:]):
        for t in np.linspace(0, 1, 26)[:-1]:
            dense.append(a + t * (b - a))
    dense = np.array(dense, dtype=int)
    approx = approximate_polygon(dense, epsilon=2.0)
    assert len(approx) == 4


def test_rdp_circle_keeps_many_vertices():
    yy, xx = np.mgrid[0:101, 0:101]
    mask = (xx - 50) ** 2 + (yy - 50) ** 2 <= 30**2
    pts = find_contours(mask)[0].points
    approx = approximate_polygon(pts, epsilon=0.02 * polygon_perimeter(pts))
    assert len(approx) >= 7


def test_convex_hull_square():
    pts = np.array([[0, 0], [5, 0], [10, 0], [10, 10], [5, 5], [0, 10]])
    hull = convex_hull(pts)
    assert len(hull) == 4
    assert polygon_area(hull) == pytest.approx(100.0)


def test_extract_features_square():
    mask = np.zeros((100, 100), dtype=bool)
    mask[20:80, 20:80] = True
    contour = find_contours(mask)[0]
    feats = extract_features(contour.points, mask, contour.area, contour.centroid)
    assert feats.n_vertices == 4
    assert feats.solidity == pytest.approx(1.0, abs=0.02)
    assert 0.75 <= feats.extent <= 1.0
    assert 0.9 <= feats.elongation <= 1.1
    assert 0.7 <= feats.circularity <= 0.85


def test_extract_features_rectangle_elongated():
    mask = np.zeros((100, 200), dtype=bool)
    mask[20:80, 20:180] = True  # 60 x 160 rectangle
    contour = find_contours(mask)[0]
    feats = extract_features(contour.points, mask, contour.area, contour.centroid)
    assert feats.n_vertices == 4
    assert feats.elongation > 1.4


def test_hu_moments_translation_invariant():
    img, _ = make_shape_image("pentagon", seed=3)
    base = np.asarray(img) < 128
    shifted = np.roll(np.roll(base, 17, axis=0), -23, axis=1)
    diff = np.mean(np.abs(log_hu_moments(base) - log_hu_moments(shifted)))
    assert diff < 1e-9


def test_hu_moments_rotation_invariant():
    # A rectangle is the right probe here: its higher Hu moments are
    # exactly zero by symmetry, so the log transform has no near-zero
    # component to flip sign.  (A triangle's h[5] is ~1e-11 and its log
    # is unstable under resampling — see the capped distance in
    # classify.py; that is a property of log-scaling, not of the
    # moments themselves.)
    img, _ = make_shape_image("rectangle", seed=3)
    rotated = img.rotate(45, resample=Image.BILINEAR, fillcolor=255)
    diff = np.mean(
        np.abs(log_hu_moments(np.asarray(img) < 128) - log_hu_moments(np.asarray(rotated) < 128))
    )
    assert diff < 0.6


def test_hu_moments_scale_invariant():
    img, _ = make_shape_image("square", seed=3)
    small = img.resize((128, 128), resample=Image.BILINEAR)
    diff = np.mean(
        np.abs(log_hu_moments(np.asarray(img) < 128) - log_hu_moments(np.asarray(small) < 128))
    )
    assert diff < 0.6


def test_hu_moments_seven_values():
    mask = np.zeros((50, 50), dtype=bool)
    mask[10:40, 10:40] = True
    assert hu_moments(mask).shape == (7,)
    assert log_hu_moments(mask).shape == (7,)
    assert np.all(np.isfinite(log_hu_moments(mask)))
