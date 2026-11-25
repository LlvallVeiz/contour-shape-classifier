"""Rule-based shape classification from contour features.

The classifier is a transparent scoring function, not a black box: every
candidate class gets a score in ``[0, 1]`` built from four interpretable
terms, and the winning class and its score are reported together with
the runner-up margin.  No training data and no fitted weights are
involved — the class prototypes are geometric facts (a square has
circularity ``pi/4``; a regular pentagon ``~0.865``) plus Hu-moment
templates measured once from the project's own synthetic generator.

Scoring (weights sum to 1)
---------------------------
* **vertices (0.45)** — the RDP vertex count at two tolerances
  (``epsilon = 0.02`` and ``0.008`` of the perimeter).  A circle's
  vertex count grows quickly as the tolerance tightens, while a true
  pentagon stays at 5 — that is what separates a noisy circle from a
  pentagon when the coarse count alone is ambiguous.  For circles and
  ellipses the fine count must also *exceed* the coarse count: a
  12-corner plus sign has 12 vertices at both tolerances, a circle
  does not.
* **circularity (0.25)** — Gaussian around the class prototype.
* **elongation (0.15)** — 1.0 inside the class's plausible interval,
  Gaussian fall-off outside it (separates square/rectangle and
  circle/ellipse).
* **Hu moments (0.15)** — distance of the log-scaled Hu vector to the
  class template measured from ideal synthetic shapes.  The per-component
  distance is capped at 5.0: the higher Hu moments are ~1e-11 (near
  zero) for symmetric shapes, so their log transform flips sign under
  resampling and would otherwise dominate the distance.

**Solidity gate.** All six classes are convex, and every clean or
noisy benchmark shape keeps solidity >= 0.97.  A contour with
solidity below 0.85 is concave and therefore out of scope — it is
reported as ``"unknown"`` instead of being forced into the closest
convex class (a plus sign would otherwise score as a circle).

A contour whose best score is below ``CONFIDENCE_THRESHOLD`` (0.55) is
also reported as ``"unknown"`` instead of a forced guess.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .contours import Contour, find_contours
from .features import ShapeFeatures, approximate_polygon, extract_features

__all__ = ["CLASSES", "Classification", "class_templates", "classify_features", "analyze_mask", "CONFIDENCE_THRESHOLD", "SOLIDITY_GATE"]

CLASSES = ["triangle", "square", "rectangle", "pentagon", "circle", "ellipse"]
CONFIDENCE_THRESHOLD = 0.55
# Below this solidity a contour is concave and out of scope for the six
# (convex) classes; see the module docstring.  Measured minimum over
# the clean + noisy benchmarks is 0.976, so 0.85 has a wide margin.
SOLIDITY_GATE = 0.85
# Per-component cap on the log-Hu distance (near-zero higher moments
# flip sign under resampling; without the cap one component can
# contribute ~50 to a mean over 7).
HU_DISTANCE_CAP = 5.0

# Expected vertex counts per class.  Circles/ellipses take the minimum
# count (6) — a smooth curve keeps gaining vertices as the RDP
# tolerance tightens, which is itself a discriminating signal.
_PROTOTYPE_VERTICES = {
    "triangle": 3,
    "square": 4,
    "rectangle": 4,
    "pentagon": 5,
    "circle": 6,
    "ellipse": 6,
}

_W_VERTEX = 0.45
_W_CIRCULARITY = 0.25
_W_ELONGATION = 0.15
_W_HU = 0.15

# Plausible PCA-elongation intervals per class.  Isotropic shapes
# (triangle, square, pentagon, circle) sit at ~1.0; rectangles and
# ellipses are generated with a side/axes ratio in [1.5, 2.5], whose
# boundary-point PCA elongation lands in ~[1.4, 2.1].  A single
# prototype value would mis-score half the ratio range, so the term is
# 1.0 inside the interval and falls off as a Gaussian in log-space
# outside it.
_ELONGATION_RANGE = {
    "triangle": (1.0, 1.20),
    "square": (1.0, 1.20),
    "rectangle": (1.35, 2.20),
    "pentagon": (1.0, 1.20),
    "circle": (1.0, 1.20),
    "ellipse": (1.35, 2.20),
}


def _elongation_score(elongation: float, lo: float, hi: float) -> float:
    if lo <= elongation <= hi:
        return 1.0
    bound = lo if elongation < lo else hi
    return float(math.exp(-((math.log(max(elongation, 1e-6)) - math.log(bound)) / 0.25) ** 2))


def _gaussian(x: float, centre: float, sigma: float) -> float:
    return float(math.exp(-((x - centre) / sigma) ** 2))


def _vertex_score(n_coarse: int, n_fine: int, expected: int) -> float:
    """Vertex-count compatibility for one class.

    ``expected`` is an exact count for polygons; for circles/ellipses
    it is the minimum count (6) with the fine count expected to be >= 8.
    """
    if expected >= 6:  # circle / ellipse: "many vertices, growing with tolerance"
        coarse = 1.0 if n_coarse >= 6 else max(0.0, 1.0 - 0.25 * (6 - n_coarse))
        if n_fine >= 8 and n_fine > n_coarse:
            fine = 1.0
        elif n_fine >= 8:
            fine = 0.6  # many corners, but no growth: a polygon, not a curve
        else:
            fine = max(0.0, 1.0 - 0.20 * (8 - n_fine))
        return 0.6 * coarse + 0.4 * fine
    coarse = 1.0 if n_coarse == expected else max(0.0, 1.0 - 0.30 * abs(n_coarse - expected))
    fine = 1.0 if n_fine == expected else max(0.0, 1.0 - 0.30 * abs(n_fine - expected))
    return 0.7 * coarse + 0.3 * fine


_TEMPLATES: dict[str, dict] | None = None


def class_templates() -> dict[str, dict]:
    """Per-class templates measured from ideal synthetic shapes.

    Each template holds the mean ``circularity``, ``elongation`` and
    log-Hu vector of the class rendered clean (no noise) at two
    rotations.  The values are *measured*, not analytic: a digitised
    triangle's traced perimeter is longer than the ideal ``3s`` (the
    staircase boundary), so its circularity is ~0.55, not the analytic
    ~0.605 — calibrating on measured values is what makes the Gaussian
    terms meaningful.  Computed lazily once per process.
    """
    global _TEMPLATES
    if _TEMPLATES is not None:
        return _TEMPLATES
    from .synthetic import make_shape_image  # local import: synthetic does not import classify

    templates: dict[str, dict] = {}
    for kind in CLASSES:
        circs, elongs, hus = [], [], []
        for seed, rotation in ((12345, 0.0), (12346, 30.0)):
            img, _truth = make_shape_image(kind, size=256, seed=seed, rotation_deg=rotation)
            mask = np.asarray(img) < 128
            contours = find_contours(mask, min_area=64)
            if not contours:
                raise RuntimeError(f"template render of {kind!r} produced no contour")
            contour = contours[0]
            component = mask  # single-shape render: the whole mask is the component
            feats = extract_features(contour.points, component, contour.area, contour.centroid)
            circs.append(feats.circularity)
            elongs.append(feats.elongation)
            hus.append(feats.hu_log)
        templates[kind] = {
            "circularity": float(np.mean(circs)),
            "elongation": float(np.mean(elongs)),
            "hu_log": np.mean(hus, axis=0),
        }
    _TEMPLATES = templates
    return templates


def hu_prototypes() -> dict[str, np.ndarray]:
    """Log-Hu templates per class (see :func:`class_templates`)."""
    return {kind: tpl["hu_log"] for kind, tpl in class_templates().items()}


@dataclass
class Classification:
    """One classified contour: label, confidence, and the evidence."""

    label: str
    confidence: float
    scores: dict[str, float]
    features: ShapeFeatures
    contour: Contour
    unknown: bool = False

    def as_dict(self) -> dict:
        """JSON-friendly view."""
        return {
            "label": self.label,
            "confidence": round(self.confidence, 4),
            "scores": {k: round(v, 4) for k, v in self.scores.items()},
            "features": self.features.as_dict(),
        }


def classify_features(features: ShapeFeatures, fine_vertices: int) -> tuple[str, float, dict[str, float]]:
    """Score one contour's features against every class.

    Returns
    -------
    (label, confidence, scores)
        ``label`` is the best class, or ``"unknown"`` when the contour
        is concave (solidity gate) or the best score is below
        :data:`CONFIDENCE_THRESHOLD`; ``confidence`` is the best score;
        ``scores`` maps every class to its score.
    """
    templates = class_templates()
    scores: dict[str, float] = {}
    for name in CLASSES:
        tpl = templates[name]
        v = _vertex_score(features.n_vertices, fine_vertices, _PROTOTYPE_VERTICES[name])
        c = _gaussian(features.circularity, tpl["circularity"], 0.10)
        lo, hi = _ELONGATION_RANGE[name]
        e = _elongation_score(features.elongation, lo, hi)
        d_hu = float(np.mean(np.minimum(np.abs(features.hu_log - tpl["hu_log"]), HU_DISTANCE_CAP)))
        h = float(math.exp(-d_hu / 1.5))
        scores[name] = _W_VERTEX * v + _W_CIRCULARITY * c + _W_ELONGATION * e + _W_HU * h
    best = max(scores, key=lambda k: scores[k])
    confidence = scores[best]
    if features.solidity < SOLIDITY_GATE or confidence < CONFIDENCE_THRESHOLD:
        return "unknown", confidence, scores
    return best, confidence, scores


def analyze_mask(mask: np.ndarray, min_area: int = 64, epsilon: float = 0.02) -> list[Classification]:
    """Find and classify every contour in a binary image.

    Parameters
    ----------
    mask:
        Boolean array, ``True`` where foreground.
    min_area:
        Components smaller than this (in pixels) are ignored.
    epsilon:
        RDP tolerance as a fraction of the contour perimeter.

    Returns
    -------
    list[Classification]
        One entry per surviving contour, sorted by decreasing area.
    """
    from scipy import ndimage  # local: only needed for the component masks

    labelled, _ = ndimage.label(mask, structure=np.ones((3, 3), dtype=int))
    results: list[Classification] = []
    for contour in find_contours(mask, min_area=min_area, labelled=labelled):
        component = labelled == contour.label_id
        features = extract_features(
            contour.points, component, contour.area, contour.centroid, epsilon=epsilon
        )
        fine = approximate_polygon(contour.points, 0.008 * max(features.perimeter, 1.0))
        label, confidence, scores = classify_features(features, len(fine))
        results.append(
            Classification(
                label=label,
                confidence=confidence,
                scores=scores,
                features=features,
                contour=contour,
                unknown=(label == "unknown"),
            )
        )
    return results
