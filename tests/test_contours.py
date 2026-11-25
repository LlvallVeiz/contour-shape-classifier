"""Tests for Moore-neighbour contour tracing."""

import numpy as np

from contour_shapes.contours import find_contours, trace_boundary
from contour_shapes.features import polygon_area, polygon_perimeter


def _square_mask(size=100, lo=20, hi=80):
    mask = np.zeros((size, size), dtype=bool)
    mask[lo:hi, lo:hi] = True
    return mask


def _disk_mask(size=101, centre=50, radius=30):
    yy, xx = np.mgrid[0:size, 0:size]
    return (xx - centre) ** 2 + (yy - centre) ** 2 <= radius**2


def test_trace_square_is_closed_ring():
    mask = _square_mask()
    contours = find_contours(mask)
    assert len(contours) == 1
    pts = contours[0].points
    assert np.array_equal(pts[0], pts[-1])  # closed ring
    assert len(pts) > 100  # every boundary pixel is visited
    assert contours[0].area == 60 * 60
    assert contours[0].bbox == (20, 20, 79, 79)


def test_trace_square_geometry():
    mask = _square_mask()
    pts = find_contours(mask)[0].points
    # Boundary pixel centres span 20..79, so the polygon side is 59.
    assert abs(polygon_area(pts) - 59 * 59) < 120
    assert abs(polygon_perimeter(pts) - 4 * 59) < 8


def test_trace_disk_geometry():
    mask = _disk_mask()
    pts = find_contours(mask)[0].points
    assert abs(polygon_area(pts) - np.pi * 30**2) < 150
    assert abs(polygon_perimeter(pts) - 2 * np.pi * 30) < 12


def test_trace_disk_is_closed():
    mask = _disk_mask()
    pts = find_contours(mask)[0].points
    assert np.array_equal(pts[0], pts[-1])
    assert len(pts) > 100


def test_empty_mask_has_no_contours():
    assert find_contours(np.zeros((10, 10), dtype=bool)) == []


def test_single_pixel_contour():
    mask = np.zeros((10, 10), dtype=bool)
    mask[5, 5] = True
    contours = find_contours(mask, min_area=1)
    assert len(contours) == 1
    assert contours[0].points.shape == (1, 2)


def test_min_area_filters_speckle():
    mask = _square_mask()
    mask[0:3, 0:3] = True  # 9-pixel speck, separate component
    assert len(find_contours(mask, min_area=64)) == 1
    assert len(find_contours(mask, min_area=1)) == 2


def test_two_components_sorted_by_area():
    mask = np.zeros((100, 100), dtype=bool)
    mask[10:20, 10:20] = True  # 100 px
    mask[50:90, 50:90] = True  # 1600 px
    contours = find_contours(mask)
    assert len(contours) == 2
    assert contours[0].area == 1600
    assert contours[1].area == 100


def test_trace_boundary_direct_call():
    mask = _square_mask()
    pts = trace_boundary(mask, (20, 20))
    assert pts.shape[1] == 2
    assert np.array_equal(pts[0], pts[-1])
