"""Check the PC pipeline gives EXACTLY the paper's features, on synthetic data.

    python self_test.py

It writes a few fake subject zips (tools/make_synthetic_dipser.py), then:
1. builds the corpus the paper's way (gtao_grid.build_corpus on the flat .zipbin layout);
2. lays the same zips out like the FTP mirror (group/experiment/subject.zip) and runs
   count_labels -> extract_features -> merge_features on it;
3. checks every array is identical and the label counts agree.
Run it once on the PC after installing Python, before starting the real extraction.
"""
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

from _common import load_grid

HERE = Path(__file__).resolve().parent
PY = sys.executable


def run(*cmd):
    print("$", " ".join(str(c) for c in cmd))
    result = subprocess.run([str(c) for c in cmd], cwd=HERE, capture_output=True, text=True)
    if result.returncode != 0:
        print(result.stdout, result.stderr)
        raise SystemExit("command failed")
    return result.stdout


def main():
    tmp = Path(tempfile.mkdtemp(prefix="gtao_selftest_"))
    try:
        flat = tmp / "flat"
        run(PY, HERE.parent / "tools" / "make_synthetic_dipser.py", flat,
            "--subjects", "4", "--seconds", "60")

        G = load_grid()
        reference = G.build_corpus(flat, tmp / "work_ref", tmp / "reference.npz")

        tree = tmp / "tree"                                  # FTP-mirror layout
        for bundle in sorted(flat.glob("*.zipbin")):
            group, experiment, subject = bundle.stem.split("__")[:3]
            dest = tree / group / experiment / f"{subject}.zip"
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(bundle, dest)
        (tree / bundle.stem.split("__")[0] / "ignored_context_cam_01.zip").write_bytes(b"x")

        counts_out = run(PY, "count_labels.py", "--data", tree, "--out", tmp / "counts.csv")
        run(PY, "extract_features.py", "--data", tree, "--out", tmp / "features",
            "--work", tmp / "work_pc", "--workers", "2")
        again = run(PY, "extract_features.py", "--data", tree, "--out", tmp / "features",
                    "--work", tmp / "work_pc", "--workers", "2")
        assert "0 to extract" in again, "resume check failed"
        run(PY, "merge_features.py", "--features", tmp / "features", "--out", tmp / "merged.npz")

        merged = np.load(tmp / "merged.npz")
        problems = []
        for name in reference.files:
            a, b = reference[name], merged[name]
            same = (a.shape == b.shape and
                    (np.array_equal(a, b, equal_nan=True) if a.dtype.kind == "f"
                     else np.array_equal(a, b)))
            print(f"  {name:18s} {'OK' if same else 'DIFFERENT'} {a.shape}")
            if not same:
                problems.append(name)

        y = reference["y"]
        expected_total = np.bincount(y, minlength=3).tolist()
        total_line = [l for l in counts_out.splitlines() if l.startswith("TOTAL")][0]
        if str(expected_total) not in total_line:
            problems.append(f"count_labels {total_line} != {expected_total}")
        print(f"  count_labels      {'OK' if str(expected_total) in total_line else 'DIFFERENT'} "
              f"{expected_total}")

        if problems:
            raise SystemExit(f"SELF-TEST FAILED: {problems}")
        print("SELF-TEST PASSED: PC pipeline == paper pipeline")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
