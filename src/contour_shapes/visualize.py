"""Visualisation: annotated images for inspection.

Draws each traced contour (green outline), its RDP approximation
(yellow polygon) and a ``label confidence`` tag at the centroid, on a
copy of the source image.  Used by the CLI ``--annotate`` option and by
the example generator.  Pillow only — no matplotlib needed.
"""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw

from .classify import Classification

__all__ = ["annotate"]

_LABEL_COLORS = {
    "triangle": (220, 38, 38),
    "square": (37, 99, 235),
    "rectangle": (124, 58, 237),
    "pentagon": (217, 119, 6),
    "circle": (5, 150, 105),
    "ellipse": (219, 39, 119),
    "unknown": (100, 116, 139),
}


def annotate(image: Image.Image, results: list[Classification]) -> Image.Image:
    """Return an RGB copy of ``image`` with contours and labels drawn.

    Parameters
    ----------
    image:
        Source image (any mode; converted to RGB).
    results:
        Classifications from :func:`contour_shapes.classify.analyze_mask`.
    """
    out = image.convert("RGB")
    draw = ImageDraw.Draw(out)
    for result in results:
        pts = [(int(x), int(y)) for x, y in result.contour.points]
        color = _LABEL_COLORS.get(result.label, (100, 116, 139))
        if len(pts) >= 2:
            draw.line(pts + [pts[0]], fill=(34, 197, 94), width=2)
        approx = [(int(x), int(y)) for x, y in result.features.approx_points]
        if len(approx) >= 2:
            draw.line(approx + [approx[0]], fill=(250, 204, 21), width=1)
        cx, cy = result.contour.centroid
        tag = f"{result.label} {result.confidence:.2f}"
        tx, ty = int(cx) + 6, int(cy) - 8
        # Small white backing so the tag stays readable on dark shapes.
        box = draw.textbbox((tx, ty), tag)
        draw.rectangle(box, fill=(255, 255, 255))
        draw.text((tx, ty), tag, fill=color)
    return out
