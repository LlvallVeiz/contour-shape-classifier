"""Image loading and binarisation.

The pipeline is deliberately small: Pillow loads the file, the image is
converted to 8-bit greyscale, and Otsu's method (Otsu, 1979) picks the
threshold that maximises the between-class variance of the histogram.
Foreground is, by default, the dark side of the threshold (dark shapes on
a light background); ``invert=True`` flips that for light shapes on a dark
background.  No OpenCV is used anywhere in this package.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from .errors import EmptyImageError, ImageLoadError

__all__ = ["load_grayscale", "otsu_threshold", "binarize", "load_binary"]


def load_grayscale(path: str | Path) -> np.ndarray:
    """Load an image file as a 2-D ``uint8`` greyscale array.

    Parameters
    ----------
    path:
        Path to any image format Pillow can read (PNG, JPG, BMP, ...).

    Raises
    ------
    ImageLoadError
        If the file does not exist or Pillow cannot decode it.
    """
    path = Path(path)
    if not path.is_file():
        raise ImageLoadError(f"image file not found: {path}")
    try:
        with Image.open(path) as img:
            gray = img.convert("L")
            return np.asarray(gray, dtype=np.uint8)
    except (OSError, ValueError) as exc:  # Pillow raises UnidentifiedImageError (OSError)
        raise ImageLoadError(f"cannot read image {path}: {exc}") from exc


def otsu_threshold(gray: np.ndarray) -> int:
    """Return Otsu's threshold for an 8-bit greyscale image.

    Otsu (1979) chooses the threshold ``t`` that maximises the
    between-class variance ``w0(t) * w1(t) * (mu0(t) - mu1(t))**2`` of the
    two pixel classes split at ``t``.  The implementation works purely on
    the 256-bin histogram, so it is O(256) after histogramming.

    Parameters
    ----------
    gray:
        2-D ``uint8`` array.
    """
    if gray.ndim != 2:
        raise ValueError("otsu_threshold expects a 2-D greyscale array")
    hist = np.bincount(gray.ravel(), minlength=256).astype(np.float64)
    total = hist.sum()
    if total == 0:
        return 0
    # Cumulative sums of counts and of intensity * count.
    weight_bg = np.cumsum(hist)
    mean_bg = np.cumsum(hist * np.arange(256))
    weight_fg = total - weight_bg
    # Class means; guard the divisions by zero at the histogram ends.
    with np.errstate(invalid="ignore", divide="ignore"):
        mu_bg = np.where(weight_bg > 0, mean_bg / np.maximum(weight_bg, 1), 0.0)
        mu_fg = np.where(
            weight_fg > 0,
            (mean_bg[-1] - mean_bg) / np.maximum(weight_fg, 1),
            0.0,
        )
        between = weight_bg * weight_fg * (mu_bg - mu_fg) ** 2
    between[weight_bg == 0] = 0.0
    between[weight_fg == 0] = 0.0
    return int(np.argmax(between))


def binarize(gray: np.ndarray, threshold: int | None = None, invert: bool = False) -> np.ndarray:
    """Binarise a greyscale image with Otsu's threshold (or a given one).

    Parameters
    ----------
    gray:
        2-D ``uint8`` array.
    threshold:
        Fixed threshold in ``[0, 255]``; when ``None`` (default) Otsu's
        threshold is computed.
    invert:
        When ``False`` (default) dark pixels are foreground.  When ``True``
        bright pixels are foreground.

    Returns
    -------
    numpy.ndarray
        Boolean array, ``True`` where the pixel is foreground.
    """
    if threshold is None:
        threshold = otsu_threshold(gray)
    if invert:
        return gray > threshold
    return gray <= threshold


def load_binary(path: str | Path, threshold: int | None = None, invert: bool = False) -> tuple[np.ndarray, int]:
    """Load an image and binarise it; return ``(mask, threshold_used)``.

    Raises
    ------
    ImageLoadError
        If the file cannot be read.
    EmptyImageError
        If the binarised image contains no foreground pixel.
    """
    gray = load_grayscale(path)
    thr = threshold if threshold is not None else otsu_threshold(gray)
    mask = binarize(gray, threshold=thr, invert=invert)
    if not mask.any():
        raise EmptyImageError(f"no foreground found in {path} (threshold={thr})")
    return mask, thr
