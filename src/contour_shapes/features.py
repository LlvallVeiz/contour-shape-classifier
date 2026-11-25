"""Geometric and moment features of a contour.

Everything here is computed from scratch with NumPy:

* **Polygon approximation** — the iterative Ramer–Douglas–Peucker
  algorithm (Ramer 1972; Douglas & Peucker 1973), the same algorithm
  behind OpenCV's ``approxPolyDP``.  The vertex count of the
  approximation is the single most informative feature for this
  classifier (3 vertices = triangle, 4 = quadrilateral, ...).
* **Area** — the shoelace formula over the traced polygon.
* **Perimeter** — summed Euclidean edge lengths.
* **Circularity** — ``4 * pi * area / perimeter**2`` (1.0 for a perfect
  circle, ``pi/4`` ≈ 0.785 for a square, lower for elongated shapes).
* **Hu moments** — the seven rotation/translation/scale invariants of
  Hu (1962), computed from the filled region's central moments and
  log-scaled the way OpenCV's ``matchShapes`` does it
  (``m_i = sign(h_i) * log|h_i|``).
* **Convex hull** — Andrew's monotone chain; gives solidity
  (``area / hull_area``) and extent (``area / bbox_area``).
* **Elongation** — ``sqrt(lambda_max / lambda_min)`` of the covariance
  of the contour points (PCA); 1.0 for isotropic shapes, > 1 for
  rectangles and ellipses.  Rotation-invariant, unlike the
  bounding-box aspect ratio.
"""

from __future__ import annotations

import math

import numpy as np

__all__ = [
    "approximate_polygon",
    "polygon_area",
    "polygon_perimeter",
    "circularity",
    "convex_hull",
    "hu_moments",
    "log_hu_moments",
    "ShapeFeatures",
    "extract_features",
]


# ---------------------------------------------------------------------------
# Polygon approximation (Ramer–Douglas–Peucker)
# ---------------------------------------------------------------------------

def _point_segment_distances(points: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Perpendicular distances of ``points`` to the segment ``a``–``b``."""
    ab = b - a
    length_sq = float(ab @ ab)
    if length_sq == 0.0:
        return np.hypot(*(points - a).T)
    t = ((points - a) @ ab) / length_sq
    t = np.clip(t, 0.0, 1.0)
    closest = a + t[:, None] * ab
    return np.hypot(*(points - closest).T)


def approximate_polygon(points: np.ndarray, epsilon: float) -> np.ndarray:
    """Simplify a closed polygon with the Ramer–Douglas–Peucker algorithm.

    Parameters
    ----------
    points:
        ``(N, 2)`` array of ``(x, y)`` vertices.  A duplicated closing
        point (first == last) is tolerated and removed first.
    epsilon:
        Maximum allowed deviation of the simplified polygon from the
        original, in pixels.  Larger values give fewer vertices.

    Returns
    -------
    numpy.ndarray
        ``(M, 2)`` array of kept vertices, ``M <= N``.  Degenerate input
        (fewer than 3 distinct points) is returned unchanged.
    """
    pts = np.asarray(points, dtype=float)
    if len(pts) >= 2 and np.array_equal(pts[0], pts[-1]):
        pts = pts[:-1]
    if len(pts) < 3:
        return pts.astype(int)
    # Start the ring at the point farthest from the centroid.  The
    # closed-ring RDP below keeps its endpoints, and the trace itself
    # starts at the raster-scan first pixel — which for a rotated
    # polygon lies mid-edge, not at a corner.  Keeping that point would
    # add one spurious vertex (a rotated pentagon would report 6).
    # The farthest point from the centroid is a genuine corner for
    # polygons and an arbitrary boundary point for smooth curves
    # (where one extra vertex is harmless).
    centre = pts.mean(axis=0)
    start_idx = int(np.argmax(np.hypot(*(pts - centre).T)))
    pts = np.roll(pts, -start_idx, axis=0)
    # Close the ring explicitly: duplicate the first point at the end so
    # the RDP endpoints coincide (a closed polygon has no true
    # endpoints).  Keeping both raw endpoints would otherwise retain
    # two adjacent points at the same corner.  The duplicate is dropped
    # from the result.
    closed = np.vstack([pts, pts[0]])
    keep = np.zeros(len(closed), dtype=bool)
    keep[0] = keep[-1] = True
    stack = [(0, len(closed) - 1)]
    while stack:
        i0, i1 = stack.pop()
        if i1 - i0 < 2:
            continue
        segment = closed[i0 : i1 + 1]
        dists = _point_segment_distances(segment, closed[i0], closed[i1])
        k = int(np.argmax(dists))
        if dists[k] > epsilon:
            keep[i0 + k] = True
            stack.append((i0, i0 + k))
            stack.append((i0 + k, i1))
    result = closed[keep]
    if len(result) >= 2 and np.array_equal(result[0], result[-1]):
        result = result[:-1]
    return result.astype(int)


# ---------------------------------------------------------------------------
# Basic geometry
# ---------------------------------------------------------------------------

def polygon_area(points: np.ndarray) -> float:
    """Signed area of a polygon via the shoelace formula (absolute value)."""
    pts = np.asarray(points, dtype=float)
    if len(pts) < 3:
        return 0.0
    x, y = pts[:, 0], pts[:, 1]
    return float(0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def polygon_perimeter(points: np.ndarray) -> float:
    """Perimeter of a closed polygon (last point connects to the first)."""
    pts = np.asarray(points, dtype=float)
    if len(pts) < 2:
        return 0.0
    shifted = np.roll(pts, -1, axis=0)
    return float(np.hypot(*(shifted - pts).T).sum())


def circularity(area: float, perimeter: float) -> float:
    """``4 * pi * area / perimeter**2``; 1.0 for a perfect circle."""
    if perimeter <= 0.0 or area <= 0.0:
        return 0.0
    return float(4.0 * math.pi * area / (perimeter * perimeter))


def convex_hull(points: np.ndarray) -> np.ndarray:
    """Convex hull of a point set (Andrew's monotone chain), CCW order."""
    pts = np.unique(np.asarray(points, dtype=float), axis=0)
    if len(pts) <= 2:
        return pts
    order = np.lexsort((pts[:, 1], pts[:, 0]))
    pts = pts[order]

    def cross(o: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
        return float((a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]))

    lower: list[np.ndarray] = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper: list[np.ndarray] = []
    for p in pts[::-1]:
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    hull = np.array(lower[:-1] + upper[:-1])
    return hull


# ---------------------------------------------------------------------------
# Hu moments (from the filled region)
# ---------------------------------------------------------------------------

def _central_moments(mask: np.ndarray) -> dict[str, float]:
    """Raw spatial moments up to order 3 of a boolean mask, centred."""
    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        return {f"mu{p}{q}": 0.0 for p in range(4) for q in range(4 - p)}
    x = xs.astype(float)
    y = ys.astype(float)
    x0, y0 = x.mean(), y.mean()
    dx, dy = x - x0, y - y0
    mu = {}
    for p in range(4):
        for q in range(4 - p):
            mu[f"mu{p}{q}"] = float(np.sum((dx**p) * (dy**q)))
    return mu


def hu_moments(mask: np.ndarray) -> np.ndarray:
    """The seven Hu invariants of a filled binary region.

    Central moments are normalised as ``eta_pq = mu_pq / mu_00**((p+q)/2 + 1)``
    (Hu, 1962) and combined into the seven invariants.  The result is
    invariant to translation, rotation and uniform scaling of the shape.
    """
    mu = _central_moments(mask)
    m00 = mu["mu00"]
    if m00 == 0.0:
        return np.zeros(7)
    eta = {}
    for p in range(4):
        for q in range(4 - p):
            eta[f"e{p}{q}"] = mu[f"mu{p}{q}"] / (m00 ** ((p + q) / 2.0 + 1.0))
    e20, e02, e11 = eta["e20"], eta["e02"], eta["e11"]
    e30, e12, e21, e03 = eta["e30"], eta["e12"], eta["e21"], eta["e03"]
    h = np.empty(7)
    h[0] = e20 + e02
    h[1] = (e20 - e02) ** 2 + 4 * e11**2
    h[2] = (e30 - 3 * e12) ** 2 + (3 * e21 - e03) ** 2
    h[3] = (e30 + e12) ** 2 + (e21 + e03) ** 2
    h[4] = (e30 - 3 * e12) * (e30 + e12) * ((e30 + e12) ** 2 - 3 * (e21 + e03) ** 2) + (
        3 * e21 - e03
    ) * (e21 + e03) * (3 * (e30 + e12) ** 2 - (e21 + e03) ** 2)
    h[5] = (e20 - e02) * ((e30 + e12) ** 2 - (e21 + e03) ** 2) + 4 * e11 * (e30 + e12) * (e21 + e03)
    h[6] = (3 * e21 - e03) * (e30 + e12) * ((e30 + e12) ** 2 - 3 * (e21 + e03) ** 2) - (
        e30 - 3 * e12
    ) * (e21 + e03) * (3 * (e30 + e12) ** 2 - (e21 + e03) ** 2)
    return h


def log_hu_moments(mask: np.ndarray) -> np.ndarray:
    """Log-scaled Hu moments: ``sign(h_i) * log|h_i|`` (natural log).

    Raw Hu values span many orders of magnitude, so distances between
    them are only meaningful after this transform — the same convention
    OpenCV's ``matchShapes`` uses for its I1/I2/I3 metrics.
    """
    h = hu_moments(mask)
    out = np.zeros(7)
    nonzero = h != 0
    out[nonzero] = np.sign(h[nonzero]) * np.log(np.abs(h[nonzero]))
    return out


# ---------------------------------------------------------------------------
# Feature bundle
# ---------------------------------------------------------------------------

class ShapeFeatures:
    """All features extracted for one contour (plain data holder)."""

    def __init__(
        self,
        area: float,
        pixel_area: int,
        perimeter: float,
        circularity_value: float,
        n_vertices: int,
        approx_points: np.ndarray,
        solidity: float,
        extent: float,
        elongation: float,
        bbox: tuple[int, int, int, int],
        centroid: tuple[float, float],
        hu_log: np.ndarray,
    ) -> None:
        self.area = area
        self.pixel_area = pixel_area
        self.perimeter = perimeter
        self.circularity = circularity_value
        self.n_vertices = n_vertices
        self.approx_points = approx_points
        self.solidity = solidity
        self.extent = extent
        self.elongation = elongation
        self.bbox = bbox
        self.centroid = centroid
        self.hu_log = hu_log

    def as_dict(self) -> dict:
        """JSON-friendly view (rounded floats, lists instead of arrays)."""
        return {
            "area": round(self.area, 2),
            "pixel_area": self.pixel_area,
            "perimeter": round(self.perimeter, 2),
            "circularity": round(self.circularity, 4),
            "n_vertices": self.n_vertices,
            "approx_points": [[int(x), int(y)] for x, y in self.approx_points],
            "solidity": round(self.solidity, 4),
            "extent": round(self.extent, 4),
            "elongation": round(self.elongation, 4),
            "bbox": list(self.bbox),
            "centroid": [round(self.centroid[0], 2), round(self.centroid[1], 2)],
            "hu_log": [round(float(v), 4) for v in self.hu_log],
        }


def extract_features(
    points: np.ndarray,
    mask: np.ndarray,
    pixel_area: int,
    centroid: tuple[float, float],
    epsilon: float = 0.02,
) -> ShapeFeatures:
    """Compute the full feature set for one traced contour.

    Parameters
    ----------
    points:
        Traced boundary points ``(N, 2)`` of ``(x, y)``.
    mask:
        Boolean mask of the component's filled pixels (same image size).
    pixel_area:
        Foreground pixel count of the component.
    centroid:
        ``(x, y)`` centre of mass of the component's pixels.
    epsilon:
        RDP tolerance as a fraction of the contour perimeter (the
        OpenCV ``approxPolyDP`` convention of ``epsilon * arcLength``).
    """
    area = polygon_area(points)
    perimeter = polygon_perimeter(points)
    approx = approximate_polygon(points, epsilon * max(perimeter, 1.0))
    hull = convex_hull(points)
    hull_area = polygon_area(hull)
    x_min, y_min, x_max, y_max = (
        int(points[:, 0].min()),
        int(points[:, 1].min()),
        int(points[:, 0].max()),
        int(points[:, 1].max()),
    )
    bbox_area = max((x_max - x_min + 1) * (y_max - y_min + 1), 1)
    # PCA elongation of the boundary-point cloud.
    centred = np.asarray(points, dtype=float)
    centred = centred - centred.mean(axis=0)
    cov = np.cov(centred.T)
    eig = np.linalg.eigvalsh(cov)
    eig = np.maximum(eig, 1e-12)
    elongation = float(math.sqrt(eig[-1] / eig[0]))
    return ShapeFeatures(
        area=area,
        pixel_area=pixel_area,
        perimeter=perimeter,
        circularity_value=circularity(area, perimeter),
        n_vertices=len(approx),
        approx_points=approx,
        solidity=float(area / hull_area) if hull_area > 0 else 0.0,
        extent=float(area / bbox_area),
        elongation=elongation,
        bbox=(x_min, y_min, x_max, y_max),
        centroid=centroid,
        hu_log=log_hu_moments(mask),
    )
