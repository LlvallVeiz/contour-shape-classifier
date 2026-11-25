"""Error types for contour-shape-classifier.

All public functions raise :class:`ContourShapesError` (or one of its
subclasses) instead of bare ``ValueError``/``RuntimeError`` so callers and
the CLI can distinguish user/input problems from internal bugs.
"""

from __future__ import annotations


class ContourShapesError(Exception):
    """Base error for this package."""


class ImageLoadError(ContourShapesError):
    """Raised when an image file cannot be read or decoded."""


class EmptyImageError(ContourShapesError):
    """Raised when binarisation finds no foreground at all."""


class NoContourError(ContourShapesError):
    """Raised when no contour survives the size filter."""


class InvalidShapeError(ContourShapesError):
    """Raised when a requested synthetic shape name is unknown."""
