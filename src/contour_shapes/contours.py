"""Contour extraction from binary images.

Two stages:

1. **Component labelling** — ``scipy.ndimage.label`` with 8-connectivity
   splits the foreground into connected components.
2. **Boundary tracing** — each component's outer boundary is walked with
   Moore-neighbour tracing (8-connected, Jacob's stopping criterion),
   the same family of border following that Suzuki & Abe (1985) formalised
   and that OpenCV's ``findContours`` implements.  The walk keeps the
   foreground on a consistent side, so the returned points form one
   closed, ordered polygon per component.

Only external (outer) contours are returned; holes inside a shape are
ignored for classification (see ``docs/RESEARCH.md`` — "Limitations").
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage

__all__ = ["Contour", "find_contours", "trace_boundary"]

# Neighbour offsets in clockwise order starting at East.  Image
# coordinates: row grows downwards, column grows rightwards, so this
# sequence is clockwise on screen: E, SE, S, SW, W, NW, N, NE.
_NEIGHBOURS = np.array(
    [
        (0, 1),  # 0: E
        (1, 1),  # 1: SE
        (1, 0),  # 2: S
        (1, -1),  # 3: SW
        (0, -1),  # 4: W
        (-1, -1),  # 5: NW
        (-1, 0),  # 6: N
        (-1, 1),  # 7: NE
    ],
    dtype=int,
)


@dataclass
class Contour:
    """One traced outer boundary.

    Attributes
    ----------
    points:
        Ordered boundary pixels as an ``(N, 2)`` integer array of
        ``(x, y)`` pairs (x = column, y = row).  The polygon is closed:
        the last point connects back to the first.
    area:
        Foreground pixel count of the component this contour bounds.
    bbox:
        Inclusive bounding box ``(x_min, y_min, x_max, y_max)``.
    centroid:
        ``(x, y)`` centre of mass of the component's pixels.
    label_id:
        Component label in the ``scipy.ndimage.label`` output for the
        image this contour came from (0 when unknown).
    """

    points: np.ndarray
    area: int
    bbox: tuple[int, int, int, int]
    centroid: tuple[float, float]
    label_id: int = 0


def trace_boundary(mask: np.ndarray, start: tuple[int, int]) -> np.ndarray:
    """Trace the outer boundary of one component with Moore tracing.

    Parameters
    ----------
    mask:
        Boolean array; ``True`` marks foreground.  It is padded internally
        so the trace never has to check image bounds.
    start:
        ``(row, col)`` of a boundary pixel of the component — a foreground
        pixel with at least one background 4-neighbour (the raster-scan
        first pixel of a component always qualifies).

    Returns
    -------
    numpy.ndarray
        ``(N, 2)`` integer array of ``(x, y)`` boundary points in trace
        order.  A single isolated pixel returns ``[[x, y]]``.
    """
    padded = np.pad(mask, 1, constant_values=False)
    r, c = start[0] + 1, start[1] + 1
    if not padded[r, c]:
        raise ValueError("start pixel is not foreground")

    # The walk pretends it arrived at the start pixel from the west
    # (which is background for a raster-scan start pixel), so the first
    # neighbour search begins just clockwise past west.
    backtrack = 4  # W
    points: list[tuple[int, int]] = [(c - 1, r - 1)]  # (x, y) in original coords
    second: tuple[int, int] | None = None

    max_steps = 8 * int(mask.sum()) + 16
    cur_r, cur_c = r, c
    for _ in range(max_steps):
        found: tuple[int, int] | None = None
        found_dir = -1
        for k in range(1, 9):
            d = (backtrack + k) % 8
            nr = cur_r + int(_NEIGHBOURS[d, 0])
            nc = cur_c + int(_NEIGHBOURS[d, 1])
            if padded[nr, nc]:
                found = (nr, nc)
                found_dir = d
                break
        if found is None:
            # Isolated pixel: no foreground neighbour at all.
            break
        nr, nc = found
        nxt = (nc - 1, nr - 1)  # (x, y) in original coords
        if second is None:
            second = nxt
            points.append(nxt)
        elif (cur_r, cur_c) == (r, c) and nxt == second:
            # Jacob's stopping criterion: back at the start pixel and
            # about to re-enter the second pixel of the walk.  The
            # start pixel was already appended on the previous step, so
            # the ring is closed (first point == last point).
            break
        else:
            points.append(nxt)
        # The backtrack direction for the next step is the direction
        # from the new current pixel back to the old one.
        backtrack = (found_dir + 4) % 8
        cur_r, cur_c = nr, nc
    else:  # pragma: no cover - safety net; max_steps is generous
        raise RuntimeError("boundary trace did not terminate")

    return np.array(points, dtype=int)


def find_contours(
    mask: np.ndarray, min_area: int = 64, labelled: np.ndarray | None = None
) -> list[Contour]:
    """Find the external contour of every component in a binary image.

    Parameters
    ----------
    mask:
        Boolean array, ``True`` where foreground.
    min_area:
        Components with fewer foreground pixels than this are dropped
        (speckle/noise filter).  Default 64.
    labelled:
        Optional pre-computed ``scipy.ndimage.label`` output for
        ``mask`` (avoids labelling twice when the caller needs it too).

    Returns
    -------
    list[Contour]
        Contours sorted by decreasing component area.  Empty when the
        image has no foreground or nothing survives ``min_area``.
    """
    if mask.ndim != 2:
        raise ValueError("find_contours expects a 2-D boolean array")
    if not mask.any():
        return []
    if labelled is None:
        structure = np.ones((3, 3), dtype=int)  # 8-connectivity
        labelled, n_labels = ndimage.label(mask, structure=structure)
    else:
        n_labels = int(labelled.max())
    contours: list[Contour] = []
    for label_id in range(1, n_labels + 1):
        ys, xs = np.nonzero(labelled == label_id)
        area = int(xs.size)
        if area < min_area:
            continue
        # Raster-scan first pixel of the component: its north and west
        # neighbours cannot belong to this component, so it is a
        # boundary pixel and a valid trace start.
        first = int(np.argmin(ys * mask.shape[1] + xs))
        start = (int(ys[first]), int(xs[first]))
        points = trace_boundary(mask & (labelled == label_id), start)
        x_min, y_min = int(xs.min()), int(ys.min())
        x_max, y_max = int(xs.max()), int(ys.max())
        contours.append(
            Contour(
                points=points,
                area=area,
                bbox=(x_min, y_min, x_max, y_max),
                centroid=(float(xs.mean()), float(ys.mean())),
                label_id=label_id,
            )
        )
    contours.sort(key=lambda ct: ct.area, reverse=True)
    return contours
