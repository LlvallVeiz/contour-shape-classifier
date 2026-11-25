"""End-to-end CLI tests (run as subprocesses, like a real user)."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parent.parent / "src"


def _run(*args, cwd=None):
    env = dict(os.environ)
    env["PYTHONPATH"] = str(SRC) + os.pathsep + env.get("PYTHONPATH", "")
    return subprocess.run(
        [sys.executable, "-m", "contour_shapes.cli", *args],
        capture_output=True,
        text=True,
        env=env,
        cwd=cwd,
    )


@pytest.fixture(scope="module")
def shape_file(tmp_path_factory):
    from contour_shapes.synthetic import make_shape_image

    tmp = tmp_path_factory.mktemp("cli")
    img, _ = make_shape_image("pentagon", seed=3)
    path = tmp / "pentagon.png"
    img.save(path)
    return tmp, path


def test_cli_classify_text(shape_file):
    _tmp, path = shape_file
    proc = _run("classify", str(path))
    assert proc.returncode == 0
    assert "pentagon" in proc.stdout


def test_cli_classify_json(shape_file):
    _tmp, path = shape_file
    proc = _run("classify", str(path), "--format", "json")
    assert proc.returncode == 0
    rows = json.loads(proc.stdout)
    assert rows[0]["label"] == "pentagon"
    assert rows[0]["confidence"] > 0.5


def test_cli_classify_annotate(shape_file):
    tmp, path = shape_file
    out = tmp / "annotated.png"
    proc = _run("classify", str(path), "--annotate", str(out))
    assert proc.returncode == 0
    assert out.is_file()
    assert out.stat().st_size > 1000


def test_cli_classify_missing_file():
    proc = _run("classify", "does-not-exist.png")
    assert proc.returncode == 2
    assert "error" in proc.stderr.lower()


def test_cli_generate(tmp_path):
    proc = _run("generate", str(tmp_path), "--count", "6", "--seed", "0")
    assert proc.returncode == 0
    assert (tmp_path / "ground_truth.json").is_file()
    assert len(list(tmp_path.glob("*.png"))) == 6


def test_cli_batch(tmp_path):
    from contour_shapes.synthetic import make_dataset

    make_dataset(tmp_path, count=6, seed=0)
    proc = _run("batch", str(tmp_path), "--format", "json")
    assert proc.returncode == 0
    rows = json.loads(proc.stdout)
    assert len(rows) == 6
    assert {r["label"] for r in rows} == {
        "triangle",
        "square",
        "rectangle",
        "pentagon",
        "circle",
        "ellipse",
    }


def test_cli_batch_empty_folder(tmp_path):
    proc = _run("batch", str(tmp_path))
    assert proc.returncode == 2


def test_cli_evaluate_small():
    proc = _run("evaluate", "--count", "12", "--seed", "5")
    assert proc.returncode == 0
    assert "Accuracy: 12/12 = 1.0000" in proc.stdout
