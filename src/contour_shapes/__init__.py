"""contour-shape-classifier: shape classification from contours, from scratch.

NumPy + Pillow + SciPy only — no OpenCV anywhere.  The pipeline is
binarise (Otsu) -> Moore-neighbour border tracing (Suzuki–Abe family)
-> Ramer–Douglas–Peucker polygon approximation -> geometric + Hu-moment
features -> transparent rule-based scoring classifier.  See
``docs/RESEARCH.md`` for the theory and the sources.
"""

from __future__ import annotations

from .classify import CLASSES, Classification, analyze_mask, class_templates, classify_features
from .contours import Contour, find_contours, trace_boundary
from .errors import (
    ContourShapesError,
    EmptyImageError,
    ImageLoadError,
    InvalidShapeError,
    NoContourError,
)
from .evaluate import classification_report, confusion_matrix, format_report
from .features import (
    ShapeFeatures,
    approximate_polygon,
    circularity,
    convex_hull,
    extract_features,
    hu_moments,
    log_hu_moments,
    polygon_area,
    polygon_perimeter,
)
from .image import binarize, load_binary, load_grayscale, otsu_threshold
from .synthetic import SHAPE_NAMES, make_dataset, make_scene_image, make_shape_image
from .visualize import annotate

__version__ = "1.0.0"

__all__ = [
    "__version__",
    "CLASSES",
    "SHAPE_NAMES",
    "Classification",
    "Contour",
    "ShapeFeatures",
    "ContourShapesError",
    "EmptyImageError",
    "ImageLoadError",
    "InvalidShapeError",
    "NoContourError",
    "analyze_mask",
    "annotate",
    "approximate_polygon",
    "binarize",
    "circularity",
    "class_templates",
    "classification_report",
    "classify_features",
    "confusion_matrix",
    "convex_hull",
    "extract_features",
    "find_contours",
    "format_report",
    "hu_moments",
    "load_binary",
    "load_grayscale",
    "log_hu_moments",
    "make_dataset",
    "make_scene_image",
    "make_shape_image",
    "otsu_threshold",
    "polygon_area",
    "polygon_perimeter",
    "trace_boundary",
]
