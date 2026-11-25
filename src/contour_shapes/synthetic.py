"""Synthetic ground-truth shapes.

All test and example images in this project are drawn here with Pillow —
no external image files, no hand-labelled data.  Every generated image
ships with a ground-truth record (label, bounding box, centroid,
rotation, size) so the classifier can be evaluated honestly against
known answers.  Generation is deterministic for a fixed seed.

Shapes: ``triangle`` (equilateral), ``square``, ``rectangle`` (aspect
1.5–2.5), ``pentagon`` (regular), ``circle``, ``ellipse`` (axes ratio
1.5–2.5).  Polygons and rectangles/ellipses are randomly rotated, so the
classifier cannot cheat on axis alignment; the features it uses are
rotation-invariant by construction.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from .errors import InvalidShapeError

__all__ = [
    "SHAPE_NAMES",
    "regular_polygon_vertices",
    "draw_shape",
    "make_shape_image",
    "make_scene_image",
    "add_gaussian_noise",
    "make_dataset",
]

SHAPE_NAMES = ["triangle", "square", "rectangle", "pentagon", "circle", "ellipse"]


def regular_polygon_vertices(
    cx: float, cy: float, radius: float, n: int, rotation_deg: float = 0.0
) -> list[tuple[float, float]]:
    """Vertices of a regular ``n``-gon, centred at ``(cx, cy)``.

    The first vertex sits at ``rotation_deg`` measured from the positive
    x-axis; vertices follow counter-clockwise in image coordinates.
    """
    angles = np.deg2rad(rotation_deg) + np.arange(n) * (2 * math.pi / n)
    return [
        (cx + radius * math.cos(a), cy + radius * math.sin(a)) for a in angles
    ]


def _rotate(points: list[tuple[float, float]], cx: float, cy: float, deg: float) -> list[tuple[float, float]]:
    """Rotate points about ``(cx, cy)`` by ``deg`` degrees."""
    theta = math.radians(deg)
    cos_t, sin_t = math.cos(theta), math.sin(theta)
    return [
        (cx + (x - cx) * cos_t - (y - cy) * sin_t, cy + (x - cx) * sin_t + (y - cy) * cos_t)
        for x, y in points
    ]


def draw_shape(
    draw: ImageDraw.ImageDraw,
    kind: str,
    cx: float,
    cy: float,
    size: float,
    rotation_deg: float = 0.0,
    rng: np.random.Generator | None = None,
) -> dict:
    """Draw one filled black shape; return its ground-truth record.

    Parameters
    ----------
    draw:
        Pillow ``ImageDraw`` to draw on.
    kind:
        One of :data:`SHAPE_NAMES`.
    cx, cy:
        Centre of the shape.
    size:
        Characteristic size in pixels — circumradius for polygons and
        circles, half the long side for squares/rectangles, semi-major
        axis for ellipses.
    rotation_deg:
        Rotation of the shape about its centre.
    rng:
        Random generator used to sample the aspect ratio of rectangles
        and ellipses; when ``None`` a 2:1 ratio is used.

    Returns
    -------
    dict
        Ground truth: ``label``, ``bbox`` (``[x0, y0, x1, y1]``),
        ``centroid`` (``[x, y]``), ``rotation_deg`` and ``size_px``.
    """
    if kind not in SHAPE_NAMES:
        raise InvalidShapeError(f"unknown shape {kind!r}; expected one of {SHAPE_NAMES}")
    rng = rng if rng is not None else np.random.default_rng(0)

    if kind == "triangle":
        pts = regular_polygon_vertices(cx, cy, size, 3, rotation_deg)
        draw.polygon(pts, fill=0)
    elif kind == "square":
        half = size / math.sqrt(2.0)  # `size` is the circumradius
        pts = [(cx - half, cy - half), (cx + half, cy - half), (cx + half, cy + half), (cx - half, cy + half)]
        pts = _rotate(pts, cx, cy, rotation_deg)
        draw.polygon(pts, fill=0)
    elif kind == "rectangle":
        ratio = float(rng.uniform(1.5, 2.5))
        half_w, half_h = size, size / ratio
        pts = [
            (cx - half_w, cy - half_h),
            (cx + half_w, cy - half_h),
            (cx + half_w, cy + half_h),
            (cx - half_w, cy + half_h),
        ]
        pts = _rotate(pts, cx, cy, rotation_deg)
        draw.polygon(pts, fill=0)
    elif kind == "pentagon":
        pts = regular_polygon_vertices(cx, cy, size, 5, rotation_deg)
        draw.polygon(pts, fill=0)
    elif kind == "circle":
        draw.ellipse([cx - size, cy - size, cx + size, cy + size], fill=0)
    elif kind == "ellipse":
        ratio = float(rng.uniform(1.5, 2.5))
        a, b = size, size / ratio
        # Sample the ellipse parametrically (72 points) and rotate the
        # point set; a 72-gon is indistinguishable from a true ellipse
        # at these resolutions and keeps the drawing path identical to
        # the other shapes (one filled polygon).
        angles = np.linspace(0, 2 * math.pi, 72, endpoint=False)
        pts = [(cx + a * math.cos(t), cy + b * math.sin(t)) for t in angles]
        pts = _rotate(pts, cx, cy, rotation_deg)
        draw.polygon(pts, fill=0)
    xs = [p[0] for p in pts] if kind not in ("circle", "ellipse") else [cx - size, cx + size]
    ys = [p[1] for p in pts] if kind not in ("circle", "ellipse") else [cy - size, cy + size]
    if kind == "ellipse":
        # Bounding box of the rotated ellipse (semi-axes a, b, angle t).
        t = math.radians(rotation_deg)
        hx = math.sqrt((a * math.cos(t)) ** 2 + (b * math.sin(t)) ** 2)
        hy = math.sqrt((a * math.sin(t)) ** 2 + (b * math.cos(t)) ** 2)
        xs, ys = [cx - hx, cx + hx], [cy - hy, cy + hy]
    return {
        "label": kind,
        "bbox": [round(min(xs), 1), round(min(ys), 1), round(max(xs), 1), round(max(ys), 1)],
        "centroid": [round(cx, 1), round(cy, 1)],
        "rotation_deg": round(rotation_deg, 1),
        "size_px": round(size, 1),
    }


def add_gaussian_noise(gray: np.ndarray, sigma: float, rng: np.random.Generator) -> np.ndarray:
    """Add zero-mean Gaussian noise (``sigma`` grey levels) and clip."""
    if sigma <= 0:
        return gray
    noisy = gray.astype(np.float64) + rng.normal(0.0, sigma, size=gray.shape)
    return np.clip(noisy, 0, 255).astype(np.uint8)


def make_shape_image(
    kind: str,
    size: int = 256,
    seed: int = 0,
    noise_sigma: float = 0.0,
    rotation_deg: float | None = None,
    shape_size: float | None = None,
) -> tuple[Image.Image, dict]:
    """Render one shape centred on a white canvas with ground truth.

    Parameters
    ----------
    kind:
        One of :data:`SHAPE_NAMES`.
    size:
        Canvas width/height in pixels.
    seed:
        Random seed (controls rotation, size and noise).
    noise_sigma:
        Gaussian noise level in grey levels (0 = clean).
    rotation_deg:
        Fixed rotation; when ``None`` a random rotation is drawn.
    shape_size:
        Fixed characteristic size; when ``None`` a random size fitting
        the canvas with margin is drawn.
    """
    rng = np.random.default_rng(seed)
    img = Image.new("L", (size, size), 255)
    draw = ImageDraw.Draw(img)
    rot = float(rng.uniform(0, 360)) if rotation_deg is None else rotation_deg
    s = float(rng.uniform(0.28, 0.38) * size) if shape_size is None else shape_size
    cx = cy = size / 2.0
    truth = draw_shape(draw, kind, cx, cy, s, rotation_deg=rot, rng=rng)
    if noise_sigma > 0:
        arr = add_gaussian_noise(np.asarray(img), noise_sigma, rng)
        img = Image.fromarray(arr, mode="L")
    truth["file"] = ""
    return img, truth


def make_scene_image(
    kinds: list[str],
    size: int = 512,
    seed: int = 0,
    noise_sigma: float = 0.0,
) -> tuple[Image.Image, list[dict]]:
    """Render several non-overlapping shapes on one canvas.

    Shapes are placed on a jittered grid so they never touch; the
    returned ground-truth list is in the same order as ``kinds``.
    """
    rng = np.random.default_rng(seed)
    img = Image.new("L", (size, size), 255)
    draw = ImageDraw.Draw(img)
    n = len(kinds)
    cols = math.ceil(math.sqrt(n))
    rows = math.ceil(n / cols)
    cell = size / max(cols, rows)
    truths: list[dict] = []
    for i, kind in enumerate(kinds):
        r, c = divmod(i, cols)
        cx = (c + 0.5) * cell + float(rng.uniform(-cell * 0.08, cell * 0.08))
        cy = (r + 0.5) * cell + float(rng.uniform(-cell * 0.08, cell * 0.08))
        s = float(rng.uniform(0.26, 0.36) * cell)
        rot = float(rng.uniform(0, 360))
        truths.append(draw_shape(draw, kind, cx, cy, s, rotation_deg=rot, rng=rng))
    if noise_sigma > 0:
        arr = add_gaussian_noise(np.asarray(img), noise_sigma, rng)
        img = Image.fromarray(arr, mode="L")
    return img, truths


def make_dataset(
    outdir: str | Path,
    count: int = 60,
    seed: int = 0,
    noise_sigma: float = 0.0,
    size: int = 256,
) -> Path:
    """Write ``count`` labelled shape images + ``ground_truth.json``.

    Images cycle through the six classes in order (so every class is
    equally represented) with a per-image seed of ``seed + index``.
    Returns the path of the written ``ground_truth.json``.
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    records: list[dict] = []
    for i in range(count):
        kind = SHAPE_NAMES[i % len(SHAPE_NAMES)]
        img, truth = make_shape_image(kind, size=size, seed=seed + i, noise_sigma=noise_sigma)
        name = f"{kind}_{i:03d}.png"
        img.save(outdir / name)
        truth["file"] = name
        records.append(truth)
    gt_path = outdir / "ground_truth.json"
    gt_path.write_text(json.dumps(records, indent=2), encoding="utf-8")
    return gt_path
