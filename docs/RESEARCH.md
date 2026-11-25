# Research notes — contour-shape-classifier

How the pipeline works, why each step is the way it is, and what the
numbers in the README actually come from.  Every claim below is either
a cited source or a measurement made by running this repository's own
code (commands are given so the measurement can be repeated).

## 1. Pipeline overview

```
image -> greyscale -> Otsu threshold -> binary mask
      -> connected components (8-connectivity)
      -> Moore-neighbour border tracing  (one closed polygon per component)
      -> Ramer–Douglas–Peucker approximation
      -> features (area, perimeter, circularity, Hu moments, solidity,
         extent, PCA elongation)
      -> rule-based scoring classifier -> label + confidence
```

The design goal is *transparency*: every decision the classifier makes
can be traced back to a named feature and a named prototype.  There is
no training set and no fitted weight vector — the "model" is four
scoring terms with fixed weights (0.45 / 0.25 / 0.15 / 0.15) plus a
solidity gate.  That is a deliberate trade-off: it caps the accuracy
ceiling below a learned classifier, but every classification is
explainable and every prototype is a measured geometric fact.

## 2. Binarisation — Otsu's method

Otsu (1979) picks the threshold that maximises the between-class
variance of the grey-level histogram.  It is the right default here
because the inputs are dark shapes on a light background (strongly
bimodal histograms), and it needs no parameter.  The implementation in
`src/contour_shapes/image.py` works on the 256-bin histogram, O(256)
after histogramming.

Measured behaviour: on the synthetic benchmark the Otsu threshold
separates the modes so cleanly that Gaussian noise up to σ=50 grey
levels (on a 0–255 scale) does not change a single classification —
see §7.  The failure mode is a *low-contrast* image (shape and
background grey levels close together); for that case the CLI accepts
`--threshold` to fix the threshold by hand.

## 3. Border following — Suzuki & Abe (1985)

`cv2.findContours` retrieves contours with the border-following
algorithm of Suzuki & Abe (1985); this project re-implements the same
family of algorithm directly: connected components are labelled first
(`scipy.ndimage.label`, 8-connectivity), then each component's outer
boundary is walked with Moore-neighbour tracing using Jacob's stopping
criterion (stop when the walk returns to the start pixel and is about
to re-enter the second pixel).  The walk keeps the foreground on a
consistent side, so the result is one ordered, closed polygon per
component.

Why re-implement instead of depending on OpenCV: `cv2` is not
installed in the build environment, and the sibling project
`template-matching-finder` already established the NumPy + Pillow +
SciPy stack for this series.  The re-implementation is verified
against known geometry in `tests/test_contours.py`: a 60×60 filled
square traces to a closed ring whose shoelace area is exactly 59² =
3481 (boundary pixel centres span 20..79) and whose perimeter is
exactly 4×59 = 236; a radius-30 disk traces to area 2736 (≈3% below
π·30² = 2827, the pixel-centre polygon sits inside the true circle)
and perimeter 197.8 (≈5% above 2π·30 = 188.5, the staircase costs
length).

Only *external* contours are returned.  Holes inside a shape are
ignored for classification — a deliberate simplification, documented
as a limitation in §8.

## 4. Polygon approximation — Ramer–Douglas–Peucker

The traced boundary is a staircase of one pixel per step.  The
Ramer–Douglas–Peucker algorithm (Ramer 1972; Douglas & Peucker 1973)
reduces it to the polygon that matters: with tolerance
`epsilon = 0.02 × perimeter` (the OpenCV `approxPolyDP` convention)
a square collapses to 4 vertices, a pentagon to 5, a circle to ~8.
The vertex count is the single most informative feature in the
classifier.

Two subtleties, both found by testing and both documented here because
they are easy to reintroduce:

1. **Closed-ring endpoints.** A closed polygon has no true endpoints,
   but a textbook RDP keeps both ends of its input array.  Applied
   naively to a traced ring, that keeps the first *and* the last
   traced pixel — two adjacent points at the same corner — and every
   polygon reports one vertex too many.  The fix closes the ring
   explicitly (first point duplicated at the end) and drops the
   duplicate from the result.
2. **Start point mid-edge.** The trace starts at the raster-scan
   first pixel, which for a rotated polygon lies mid-edge, not at a
   corner — and the ring RDP keeps its start point.  A rotated
   pentagon then reported 6 vertices.  The fix rotates the ring to
   start at the point farthest from the centroid, which is a genuine
   corner for polygons (and an arbitrary boundary point for smooth
   curves, where one extra vertex is harmless).

## 5. Features

| Feature | Definition | Why it is here |
| --- | --- | --- |
| Area | shoelace formula over the traced polygon | scale (filtering, reporting) |
| Perimeter | summed Euclidean edge lengths | circularity denominator |
| Circularity | `4π·area / perimeter²` | 1.0 circle, π/4 ≈ 0.785 square, lower when elongated |
| Hu moments (7) | Hu (1962), from the filled region's central moments, log-scaled as `sign(h)·log\|h\|` (the `matchShapes` convention) | rotation/translation/scale-invariant shape signature |
| Solidity | `area / convex-hull area` | 1.0 for convex shapes; the concavity gate (§6) |
| Extent | `area / bounding-box area` | fill ratio of the bounding box |
| Elongation | `√(λmax/λmin)` of the boundary-point covariance (PCA) | rotation-invariant aspect ratio; separates square/rectangle and circle/ellipse |

**Digitised vs analytic circularity.** The analytic circularity of
an equilateral triangle is ≈ 0.605, but the *measured* value on a
rasterised triangle is ≈ 0.55: the staircase boundary is longer than
the ideal perimeter.  The classifier therefore calibrates on
*measured* prototypes, not analytic values.  The table below is the
output of `class_templates()` — the mean of two clean renders
(rotations 0° and 30°) per class:

| class | circularity (measured) | elongation (measured) |
| --- | --- | --- |
| triangle | 0.548 | 1.008 |
| square | 0.733 | 1.003 |
| rectangle | 0.684 | 1.481 |
| pentagon | 0.776 | 1.013 |
| circle | 0.898 | 1.002 |
| ellipse | 0.805 | 1.469 |

Note the rectangle/ellipse elongation is *not* the side ratio: the PCA
elongation of a boundary-point cloud is lower than the aspect ratio
(a 2:1 rectangle measures ≈ 1.87).  That is why the classifier scores
elongation against a plausible *interval* ([1.35, 2.20] for
rectangle/ellipse, [1.0, 1.20] for the isotropic classes) instead of a
single prototype value.

**Hu-moment numerics.** The higher Hu moments are ~1e-11 (near zero)
for symmetric shapes, so their log transform flips sign when the
rasterisation changes slightly (a triangle rotated 45° moves log h₅
from −27.2 to +24.7).  The classifier therefore caps the
per-component log-Hu distance at 5.0 before averaging; without the cap
one component contributes ~52 to a mean over 7 and the Hu term becomes
noise.  This is a property of log-scaling near-zero quantities, not a
bug in the moments — the raw moments themselves are stable, and the
test suite checks rotation invariance on a rectangle, where the higher
moments are exactly zero by symmetry.

## 6. Classifier

Four weighted terms (vertices 0.45, circularity 0.25, elongation
0.15, Hu 0.15) plus a **solidity gate**: solidity below 0.85 means
the shape is concave and therefore out of scope — it is reported as
`"unknown"` instead of being forced into the closest convex class (a
plus sign otherwise scores 0.75 as a circle).  The gate threshold has
a wide margin: the lowest solidity measured over the clean and noisy
benchmarks is 0.976.  A second guard is the vertex-growth check: a
circle's RDP vertex count grows when the tolerance tightens (8 → 16
at ε = 0.02 → 0.008 of the perimeter), while a 12-corner plus sign
stays at 12 — so the circle/ellipse vertex term requires the fine
count to exceed the coarse count.

Confidence is the winning class's score.  Below 0.55 the label is
`"unknown"`.

## 7. Measured results

Benchmark command (from the project root):

```console
$ contour-shapes evaluate --count 120 --seed 1000
$ contour-shapes evaluate --count 120 --seed 2000 --noise 30
```

`evaluate` renders 120 fresh images (20 per class, random rotation
and size, seed as given) and classifies them.  Results, measured
2026-11-25 on this machine (Python 3.12, NumPy 1.26, Pillow 10.2,
SciPy 1.11):

| benchmark | accuracy | mean confidence |
| --- | --- | --- |
| clean, seed 1000 | 120/120 = 1.0000 | 0.9009 |
| Gaussian noise σ=10, seed 2000 | 120/120 = 1.0000 | 0.9103 |
| Gaussian noise σ=20, seed 2000 | 120/120 = 1.0000 | 0.9103 |
| Gaussian noise σ=30, seed 2000 | 120/120 = 1.0000 | 0.8980 |
| Gaussian noise σ=50, seed 3000 | 120/120 = 1.0000 | 0.8030 |
| small canvas (128 px) + noise σ=15, 60 images | 60/60 = 1.0000 | — |
| six-shape scene, clean and σ=20 | 6/6 and 6/6 | — |

Confusion matrices are perfectly diagonal in every run above (see
`examples/RESULTS.md` for the per-example numbers).  Accuracy stays
1.0000 up to σ=50 because Otsu's threshold on a strongly bimodal
histogram is almost unaffected by additive noise; what degrades is
the *confidence* (0.90 → 0.80), which is the honest signal that the
inputs are getting harder.  The benchmark measures generalisation
across seeds, rotations, sizes and noise levels — not across drawing
styles: every image comes from the same generator, and the Hu
templates are measured from that generator too.

## 8. Limitations

* **Only convex, isolated, filled shapes.** Touching or overlapping
  shapes merge into one component; holes are ignored; concave shapes
  are reported `"unknown"` by design.
* **No OpenCV parity check.** The tracer and RDP are verified against
  hand-computed geometry (square 59², disk πr²), not against
  `cv2.findContours` output, because OpenCV is not installed here.
* **Templates come from the same generator as the benchmark.** The
  1.0000 accuracy is a statement about the generator's distribution,
  not about arbitrary photographs.  Real photos (lighting gradients,
  textured backgrounds, perspective distortion) are out of scope for
  v1.0.0.
* **Tiny shapes.** Below ~20 px across, the staircase dominates the
  geometry and vertex counts become unreliable; `min_area` (default
  64 px²) drops speckle, but a 3×3 square is still — correctly —
  classified as a square.

## 9. References

1. Suzuki, S. & Abe, K. (1985). Topological structural analysis of
   digitized binary images by border following. *Computer Vision,
   Graphics, and Image Processing*, 30(1), 32–46.
   https://doi.org/10.1016/0734-189X(85)90016-7 —
   https://www.sciencedirect.com/science/article/pii/0734189X85900167
2. Hu, M.-K. (1962). Visual pattern recognition by moment invariants.
   *IRE Transactions on Information Theory*, 8(2), 179–187.
   https://doi.org/10.1109/TIT.1962.1057692
3. Ramer, U. (1972). An iterative procedure for the polygonal
   approximation of plane curves. *Computer Graphics and Image
   Processing*, 1(3), 244–256.
   https://doi.org/10.1016/S0146-664X(72)80017-0
4. Douglas, D. H. & Peucker, T. K. (1973). Algorithms for the reduction
   of the number of points required to represent a digitized line or
   its caricature. *The Canadian Cartographer*, 10(2), 112–122.
   https://doi.org/10.3138/FM57-6770-U75U-7727
5. Otsu, N. (1979). A threshold selection method from gray-level
   histograms. *IEEE Transactions on Systems, Man, and Cybernetics*,
   9(1), 62–66. https://doi.org/10.1109/TSMC.1979.4310076
6. OpenCV documentation, "Structural Analysis and Shape Descriptors"
   (`findContours`, `approxPolyDP`, `HuMoments`, `matchShapes`).
   https://docs.opencv.org/3.4.4/d3/dc0/group__imgproc__shape.html
