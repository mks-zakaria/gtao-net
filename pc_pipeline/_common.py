"""Shared helpers for the PC pipeline (download -> count -> extract -> merge).

The feature code is NOT re-implemented here. Every script imports the v2 grid
script (``kaggle/v2_nested/dipser_grid/gtao_grid.py``) and calls its own
functions (``subject_windows``, ``read_watch_sensors``, ``encode_windows``), so
features extracted on the PC are identical to the ones the paper used.
"""
import hashlib
import importlib.util
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_GRID = HERE.parent / "kaggle" / "v2_nested" / "dipser_grid" / "gtao_grid.py"


def load_grid(grid_path=None):
    """Import gtao_grid.py from its path and return the module."""
    path = Path(grid_path or os.environ.get("GTAO_GRID", DEFAULT_GRID)).resolve()
    if not path.is_file():
        sys.exit(f"gtao_grid.py not found at {path} (set --grid or GTAO_GRID)")
    spec = importlib.util.spec_from_file_location("gtao_grid", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.__grid_path__ = str(path)
    module.__grid_sha256__ = hashlib.sha256(path.read_bytes()).hexdigest()
    return module


def find_bundles(root):
    """Return [(bundle_path, group_key, stem)] sorted by stem.

    Two layouts are accepted:
    * rclone mirror of the FTP: root/group_01/experiment_01/subject_01.zip
    * flat Kaggle layout:       root/group_01__experiment_01__subject_01.zipbin
    ``group_key`` is "group_01/experiment_01/subject_01" (the subject id used by
    the grid), ``stem`` is "group_01__experiment_01__subject_01". Sorting by stem
    reproduces the order in which gtao_grid.build_corpus reads the flat layout.
    Only finished files are listed (rclone downloads to *.partial first).
    """
    root = Path(root)
    found = {}
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix not in (".zip", ".zipbin"):
            continue
        if path.suffix == ".zipbin" or "__" in path.stem:
            parts = path.stem.split("__")[:3]
        else:
            rel = path.relative_to(root).with_suffix("").parts
            parts = list(rel[-3:])
        if len(parts) != 3 or not parts[2].startswith("subject_"):
            continue   # context_cam_NN.zip and anything else is ignored
        stem = "__".join(parts)
        found[stem] = (path, "/".join(parts), stem)
    return [found[s] for s in sorted(found)]


def feature_dims(G):
    """Fixed per-window feature length of every stream."""
    return {"face": G.FACE_K * G.FACE_K * G.SEQ_K,
            "facemesh": 478 * 3 * G.SEQ_K,
            "head_pose": 3 * G.SEQ_K,
            "physio": len(G.PHYSIO_CHANNELS) * G.SEQ_K}


def atomic_savez(path, **arrays):
    """Write an .npz so that a crash never leaves a half-written file behind."""
    import numpy as np
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as handle:
        np.savez(handle, **arrays)
    os.replace(tmp, path)
