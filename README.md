# contour-shape-classifier

Classify geometric shapes — triangle, square, rectangle, pentagon,
circle, ellipse — from image contours, **from scratch** in Python
(NumPy + Pillow + SciPy).  No OpenCV anywhere: the binarisation
(Otsu), the border tracing (Moore-neighbour, Suzuki–Abe family), the
polygon approximation (Ramer–Douglas–Peucker), the Hu moments and the
classifier itself are all implemented in this repository.

```
$ contour-shapes classify examples/shapes/pentagon.png
examples/shapes/pentagon.png:0  pentagon conf=0.89 area=18892 circ=0.774 vertices=5 elong=1.01
```

## Background

A contour is the ordered list of pixels around a shape's boundary.
Once a shape is reduced to its contour, a handful of geometric
features are enough to name it: how many corners its polygonal
approximation has (3 = triangle, 4 = quadrilateral, 5 = pentagon),
how close it is to a circle (`circularity = 4π·area/perimeter²`),
how elongated it is (PCA of the boundary points), and its seven Hu
moments — rotation-, translation- and scale-invariant signatures from
Hu (1962).  The classifier scores every candidate class on those
features and reports the winner with a confidence.  The full theory,
the measured calibration values and the benchmark results are in
[`docs/RESEARCH.md`](docs/RESEARCH.md).

## Installation

Requires Python ≥ 3.10, NumPy ≥ 1.26, Pillow ≥ 10, SciPy ≥ 1.11.
`cv2` is **not** required — and not installed in the build
environment, which is exactly why this project exists in this form.

```console
$ pip install -e .
$ contour-shapes --help
```

For development (tests):

```console
$ pip install -e ".[dev]"
$ pytest
```

## Usage

```console
# Classify one image (text, JSON or CSV output; --annotate draws the result)
$ contour-shapes classify photo.png --format json --annotate annotated.png

# Classify every PNG in a folder
$ contour-shapes batch photos/ --format csv --output results.csv

# Generate a labelled synthetic dataset (images + ground_truth.json)
$ contour-shapes generate data/ --count 60 --seed 0 --noise 15

# Benchmark on 120 fresh synthetic images (confusion matrix + metrics)
$ contour-shapes evaluate --count 120 --seed 1000
```

`classify` accepts `--min-area` (ignore components smaller than N
px², default 64), `--invert` (light shapes on a dark background) and
`--threshold` (fixed threshold instead of Otsu).  `evaluate` exits 1
if accuracy falls below `--min-accuracy` (default 0.90).

JSON output for one contour looks like this (real output):

```json
[
  {
    "file": "examples/shapes/pentagon.png",
    "contour": 0,
    "label": "pentagon",
    "confidence": 0.8901,
    "area": 18891.5,
    "pixel_area": 19129,
    "perimeter": 553.77,
    "circularity": 0.7741,
    "n_vertices": 5,
    "elongation": 1.0145,
    "solidity": 0.9889,
    "centroid_x": 127.55,
    "centroid_y": 127.68
  }
]
```

## Project structure

```text
contour-shape-classifier/
├── src/contour_shapes/
│   ├── __init__.py      public API
│   ├── cli.py           contour-shapes command (classify/batch/generate/evaluate)
│   ├── image.py         loading + Otsu binarisation
│   ├── contours.py      Moore-neighbour border tracing (Suzuki–Abe family)
│   ├── features.py      RDP approximation, geometry, Hu moments, hull, PCA
│   ├── classify.py      rule-based scoring classifier + solidity gate
│   ├── synthetic.py     synthetic ground-truth shapes + noise
│   ├── evaluate.py      confusion matrix, precision/recall/F1
│   ├── visualize.py     annotated images (contour + label tags)
│   └── errors.py        error types
├── tests/               79 pytest tests
├── examples/            generated images + results (see examples/RESULTS.md)
├── docs/RESEARCH.md     theory, calibration, benchmarks, references
├── pyproject.toml
└── requirements.txt
```

## Example output

Single shapes (`examples/shapes/`, 6/6 correct) and the six-shape
scene (`examples/scene.png`, 6/6 correct, clean and with Gaussian
noise σ=20):

```text
triangle.png:0  triangle conf=0.87 area=8418 circ=0.537 vertices=3 elong=1.01
square.png:0  square conf=0.90 area=14927 circ=0.690 vertices=4 elong=1.00
rectangle.png:0  rectangle conf=0.93 area=14265 circ=0.625 vertices=4 elong=1.48
pentagon.png:0  pentagon conf=0.89 area=18892 circ=0.774 vertices=5 elong=1.01
circle.png:0  circle conf=1.00 area=29863 circ=0.898 vertices=8 elong=1.00
ellipse.png:0  ellipse conf=1.00 area=10166 circ=0.815 vertices=8 elong=1.41
```

Benchmark (`contour-shapes evaluate --count 120`, measured
2026-11-25):

| benchmark | accuracy | mean confidence |
| --- | --- | --- |
| clean, seed 1000 | 120/120 = 1.0000 | 0.9009 |
| noise σ=10 / 20 / 30 | 120/120 = 1.0000 | 0.9103 / 0.9103 / 0.8980 |
| noise σ=50 | 120/120 = 1.0000 | 0.8030 |
| small canvas (128 px) + noise σ=15 | 60/60 = 1.0000 | — |

Confusion matrices are perfectly diagonal in every run.  See
[`examples/RESULTS.md`](examples/RESULTS.md) for the per-example
numbers and [`docs/RESEARCH.md`](docs/RESEARCH.md) for why accuracy
holds while confidence degrades with noise.

## Design decisions

* **No OpenCV.** `cv2` is not installed in the build environment;
  every algorithm is re-implemented and verified against
  hand-computed geometry (a 60×60 square traces to area exactly
  3481 = 59² and perimeter exactly 236 = 4×59).
* **Transparent scoring, not a learned model.** Four interpretable
  terms (vertices 0.45, circularity 0.25, elongation 0.15, Hu 0.15)
  plus a solidity gate.  Every classification can be explained from
  its features; there is no training set to overfit.
* **Measured prototypes, not analytic ones.** A rasterised triangle
  measures circularity ≈ 0.55, not the analytic ≈ 0.605, because the
  staircase boundary is longer than the ideal perimeter — so the
  class templates are measured from the project's own generator.
* **Solidity gate for out-of-scope shapes.** All six classes are
  convex; a concave shape (plus sign, L-shape) is reported
  `"unknown"` instead of being forced into the closest convex class.
  The gate (0.85) has a wide margin — the lowest solidity measured
  over the benchmarks is 0.976.

## Limitations

* Only convex, isolated, filled shapes.  Touching/overlapping shapes
  merge into one component; holes are ignored; concave shapes are
  `"unknown"` by design.
* No OpenCV parity check — the tracer and RDP are verified against
  hand-computed geometry, not against `cv2.findContours`, because
  OpenCV is not installed here.
* Templates and benchmarks come from the same synthetic generator, so
  the 1.0000 accuracy is a statement about that distribution, not
  about arbitrary photographs.  Real photos (lighting gradients,
  textured backgrounds, perspective) are out of scope for v1.0.0.
* Below ~20 px across, the staircase dominates the geometry and
  vertex counts become unreliable.

## Tests

```console
$ pytest            # 79 tests, ~13 s
```

Coverage: Otsu binarisation, border tracing (square/disk geometry,
closed rings, speckle filter), RDP (line → 2 points, square → 4,
circle → ≥7), Hu-moment invariances, all six classes clean/rotated/
noisy, concave-shape `"unknown"`, multi-shape scenes, evaluation
metrics, and the CLI end to end (classify/batch/generate/evaluate).

## License

MIT — see [LICENSE](LICENSE).
