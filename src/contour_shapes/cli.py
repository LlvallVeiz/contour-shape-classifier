"""Command-line interface: ``contour-shapes``.

Subcommands
-----------
``classify``
    Classify the shapes in one or more images.
``batch``
    Classify every image in a folder (optionally recursive) and write a
    CSV/JSON summary.
``generate``
    Write a labelled synthetic dataset (images + ``ground_truth.json``).
``evaluate``
    Run the classifier on a fresh synthetic benchmark and print the
    confusion matrix, accuracy and per-class precision/recall/F1.

Exit codes: 0 = success, 1 = evaluation accuracy below the requested
minimum, 2 = usage/input error (missing file, empty folder, ...).
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

from . import __version__
from .classify import CLASSES, analyze_mask
from .errors import ContourShapesError
from .evaluate import classification_report, format_report
from .image import load_binary, load_grayscale, otsu_threshold
from .synthetic import SHAPE_NAMES, make_dataset, make_scene_image, make_shape_image
from .visualize import annotate

__all__ = ["main", "build_parser"]


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _analyse_file(path: Path, min_area: int, invert: bool, threshold: int | None):
    """Load one image file and classify its contours."""
    mask, thr = load_binary(path, threshold=threshold, invert=invert)
    return analyze_mask(mask, min_area=min_area), thr


def _result_rows(path: Path, results) -> list[dict]:
    rows = []
    for i, res in enumerate(results):
        f = res.features
        rows.append(
            {
                "file": str(path),
                "contour": i,
                "label": res.label,
                "confidence": round(res.confidence, 4),
                "area": round(f.area, 2),
                "pixel_area": f.pixel_area,
                "perimeter": round(f.perimeter, 2),
                "circularity": round(f.circularity, 4),
                "n_vertices": f.n_vertices,
                "elongation": round(f.elongation, 4),
                "solidity": round(f.solidity, 4),
                "centroid_x": round(f.centroid[0], 2),
                "centroid_y": round(f.centroid[1], 2),
            }
        )
    return rows


def _emit(rows: list[dict], fmt: str, output: Path | None) -> None:
    """Write rows as text/JSON/CSV to stdout or a file."""
    if fmt == "json":
        text = json.dumps(rows, indent=2)
    elif fmt == "csv":
        buf = io.StringIO()
        if rows:
            writer = csv.DictWriter(buf, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        text = buf.getvalue()
    else:  # text
        lines = []
        for row in rows:
            lines.append(
                f"{row['file']}:{row['contour']}  {row['label']} "
                f"conf={row['confidence']:.2f} area={row['area']:.0f} "
                f"circ={row['circularity']:.3f} vertices={row['n_vertices']} "
                f"elong={row['elongation']:.2f}"
            )
        text = "\n".join(lines)
    if output is not None:
        output.write_text(text + ("\n" if text and not text.endswith("\n") else ""), encoding="utf-8")
    elif text:
        print(text)


# ---------------------------------------------------------------------------
# Subcommands
# ---------------------------------------------------------------------------

def cmd_classify(args: argparse.Namespace) -> int:
    all_rows: list[dict] = []
    annotate_targets: list[tuple[Path, list]] = []
    for raw in args.images:
        path = Path(raw)
        try:
            results, _thr = _analyse_file(path, args.min_area, args.invert, args.threshold)
        except ContourShapesError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
        if not results:
            print(f"{path}: no contour survived min_area={args.min_area}", file=sys.stderr)
        all_rows.extend(_result_rows(path, results))
        if args.annotate is not None:
            annotate_targets.append((path, results))
    _emit(all_rows, args.format, args.output)
    if args.annotate is not None and annotate_targets:
        if len(annotate_targets) > 1:
            print("error: --annotate accepts a single input image", file=sys.stderr)
            return 2
        path, results = annotate_targets[0]
        with Image.open(path) as img:
            annotate(img, results).save(args.annotate)
        print(f"annotated image written to {args.annotate}", file=sys.stderr)
    return 0


def cmd_batch(args: argparse.Namespace) -> int:
    folder = Path(args.folder)
    if not folder.is_dir():
        print(f"error: not a directory: {folder}", file=sys.stderr)
        return 2
    pattern = args.pattern
    files = sorted(folder.rglob(pattern) if args.recursive else folder.glob(pattern))
    files = [f for f in files if f.is_file()]
    if not files:
        print(f"error: no files matching {pattern!r} in {folder}", file=sys.stderr)
        return 2
    all_rows: list[dict] = []
    failed = 0
    for path in files:
        try:
            results, _thr = _analyse_file(path, args.min_area, args.invert, args.threshold)
        except ContourShapesError as exc:
            print(f"warning: skipping {path}: {exc}", file=sys.stderr)
            failed += 1
            continue
        all_rows.extend(_result_rows(path, results))
    _emit(all_rows, args.format, args.output)
    print(f"{len(files) - failed}/{len(files)} files processed", file=sys.stderr)
    return 0


def cmd_generate(args: argparse.Namespace) -> int:
    outdir = Path(args.outdir)
    gt_path = make_dataset(outdir, count=args.count, seed=args.seed, noise_sigma=args.noise, size=args.size)
    print(f"wrote {args.count} images to {outdir} (+ {gt_path.name})")
    return 0


def cmd_evaluate(args: argparse.Namespace) -> int:
    pairs: list[tuple[str, str]] = []
    confidences: list[float] = []
    for i in range(args.count):
        kind = SHAPE_NAMES[i % len(SHAPE_NAMES)]
        img, _truth = make_shape_image(kind, size=args.size, seed=args.seed + i, noise_sigma=args.noise)
        mask = np.asarray(img) < 128
        results = analyze_mask(mask, min_area=args.min_area)
        predicted = results[0].label if results else "unknown"
        pairs.append((kind, predicted))
        confidences.append(results[0].confidence if results else 0.0)
    report = classification_report(pairs)
    if args.format == "json":
        payload = dict(report)
        payload["mean_confidence"] = round(float(np.mean(confidences)), 4) if confidences else 0.0
        payload["pairs"] = pairs
        print(json.dumps(payload, indent=2))
    else:
        print(format_report(pairs))
        print(f"\nMean confidence: {float(np.mean(confidences)):.4f}")
        print(f"Benchmark: {args.count} images, noise sigma={args.noise}, seed={args.seed}")
    return 0 if report["accuracy"] >= args.min_accuracy else 1


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="contour-shapes",
        description="Classify geometric shapes from image contours (NumPy + Pillow, no OpenCV).",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    p_classify = sub.add_parser("classify", help="classify shapes in one or more images")
    p_classify.add_argument("images", nargs="+", help="image file(s) to classify")
    p_classify.add_argument("--min-area", type=int, default=64, help="ignore components smaller than this (px^2)")
    p_classify.add_argument("--invert", action="store_true", help="light shapes on a dark background")
    p_classify.add_argument("--threshold", type=int, default=None, help="fixed threshold instead of Otsu")
    p_classify.add_argument("--format", choices=["text", "json", "csv"], default="text")
    p_classify.add_argument("--output", type=Path, default=None, help="write results to this file")
    p_classify.add_argument("--annotate", type=Path, default=None, help="write an annotated PNG (single image only)")
    p_classify.set_defaults(func=cmd_classify)

    p_batch = sub.add_parser("batch", help="classify every image in a folder")
    p_batch.add_argument("folder", help="folder with images")
    p_batch.add_argument("--recursive", action="store_true")
    p_batch.add_argument("--pattern", default="*.png", help="glob pattern (default *.png)")
    p_batch.add_argument("--min-area", type=int, default=64)
    p_batch.add_argument("--invert", action="store_true")
    p_batch.add_argument("--threshold", type=int, default=None)
    p_batch.add_argument("--format", choices=["text", "json", "csv"], default="csv")
    p_batch.add_argument("--output", type=Path, default=None)
    p_batch.set_defaults(func=cmd_batch)

    p_generate = sub.add_parser("generate", help="write a labelled synthetic dataset")
    p_generate.add_argument("outdir", help="output folder")
    p_generate.add_argument("--count", type=int, default=60)
    p_generate.add_argument("--seed", type=int, default=0)
    p_generate.add_argument("--noise", type=float, default=0.0, help="Gaussian noise sigma (grey levels)")
    p_generate.add_argument("--size", type=int, default=256)
    p_generate.set_defaults(func=cmd_generate)

    p_evaluate = sub.add_parser("evaluate", help="benchmark on a fresh synthetic dataset")
    p_evaluate.add_argument("--count", type=int, default=120)
    p_evaluate.add_argument("--seed", type=int, default=0)
    p_evaluate.add_argument("--noise", type=float, default=0.0)
    p_evaluate.add_argument("--size", type=int, default=256)
    p_evaluate.add_argument("--min-area", type=int, default=64)
    p_evaluate.add_argument("--min-accuracy", type=float, default=0.90, help="exit 1 below this accuracy")
    p_evaluate.add_argument("--format", choices=["text", "json"], default="text")
    p_evaluate.set_defaults(func=cmd_evaluate)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
