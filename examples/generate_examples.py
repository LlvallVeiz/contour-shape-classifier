"""Regenerate every example in this folder from scratch.

Run from the project root::

    python examples/generate_examples.py

Outputs (all regenerated deterministically — fixed seeds):

* ``shapes/<kind>.png`` — one clean image per class + ``shapes/ground_truth.json``
* ``scene.png`` — six shapes on one canvas + ``scene_ground_truth.json``
* ``scene_noisy.png`` — the same scene with Gaussian noise (sigma=20)
* ``output/<name>.{json,csv,txt}`` — classifier results per example
* ``output/scene_annotated.png`` — the scene with contours + labels drawn

The numbers quoted in ``examples/RESULTS.md`` and ``README.md`` come
from running this script; re-run it after any classifier change and
update those files if the numbers move.
"""

from __future__ import annotations

import csv
import io
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from contour_shapes import analyze_mask, annotate  # noqa: E402
from contour_shapes.image import load_binary  # noqa: E402
from contour_shapes.synthetic import (  # noqa: E402
    SHAPE_NAMES,
    make_scene_image,
    make_shape_image,
)

ROOT = Path(__file__).resolve().parent


def _write_results(name: str, rows: list[dict]) -> None:
    outdir = ROOT / "output"
    outdir.mkdir(exist_ok=True)
    (outdir / f"{name}.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    buf = io.StringIO()
    if rows:
        writer = csv.DictWriter(buf, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    (outdir / f"{name}.csv").write_text(buf.getvalue(), encoding="utf-8")
    lines = [
        f"{r['file']}:{r['contour']}  {r['label']} conf={r['confidence']:.2f} "
        f"area={r['area']:.0f} circ={r['circularity']:.3f} vertices={r['n_vertices']} "
        f"elong={r['elongation']:.2f}"
        for r in rows
    ]
    (outdir / f"{name}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _rows_for(path: Path) -> tuple[list[dict], list, Image.Image]:
    mask, _thr = load_binary(path)
    results = analyze_mask(mask)
    with Image.open(path) as img:
        image = img.copy()
    rows = []
    for i, res in enumerate(results):
        f = res.features
        rows.append(
            {
                "file": path.name,
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
    return rows, results, image


def main() -> int:
    shapes_dir = ROOT / "shapes"
    shapes_dir.mkdir(exist_ok=True)
    (ROOT / "output").mkdir(exist_ok=True)

    # 1. One clean image per class.
    truths = []
    for i, kind in enumerate(SHAPE_NAMES):
        img, truth = make_shape_image(kind, seed=101 + i)
        name = f"{kind}.png"
        img.save(shapes_dir / name)
        truth["file"] = name
        truths.append(truth)
    (shapes_dir / "ground_truth.json").write_text(json.dumps(truths, indent=2), encoding="utf-8")

    # 2. Six-shape scene, clean and noisy.
    kinds = ["triangle", "square", "rectangle", "pentagon", "circle", "ellipse"]
    scene, scene_truths = make_scene_image(kinds, size=512, seed=99)
    scene.save(ROOT / "scene.png")
    (ROOT / "scene_ground_truth.json").write_text(
        json.dumps(scene_truths, indent=2), encoding="utf-8"
    )
    noisy, _ = make_scene_image(kinds, size=512, seed=99, noise_sigma=20.0)
    noisy.save(ROOT / "scene_noisy.png")

    # 3. Classify everything and write the result files.
    all_rows: list[dict] = []
    for name in [f"{kind}.png" for kind in SHAPE_NAMES]:
        rows, _res, _img = _rows_for(shapes_dir / name)
        all_rows.extend(rows)
    _write_results("shapes", all_rows)

    rows, results, image = _rows_for(ROOT / "scene.png")
    _write_results("scene", rows)
    annotate(image, results).save(ROOT / "output" / "scene_annotated.png")

    rows_noisy, _res, _img = _rows_for(ROOT / "scene_noisy.png")
    _write_results("scene_noisy", rows_noisy)

    n_ok = sum(1 for r in all_rows if r["label"] == r["file"].split(".")[0])
    print(f"single shapes: {n_ok}/{len(all_rows)} correct")
    print(f"scene: {[r['label'] for r in rows]}")
    print(f"scene (noisy sigma=20): {[r['label'] for r in rows_noisy]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
