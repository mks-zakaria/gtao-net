"""Phase 7: face-crop moments in several discrete orthogonal families, per subject zip.

Why: the paper's "face" stream is the WHOLE 640x480 webcam frame resized to 64x64 (the
scene: classroom, whiteboard, classmates), not a face. This extractor adds a face_crop
stream: each frame is cropped to the student's own face box (metadata person.face
.bounding_box in dict form {x0,y0,x1,y1}; the list form marks small far-away detections and
carries no facemesh, so it is ignored), squared with a 10% margin, grey, 64x64, and
projected on 5 moment families (moment_families.FAMILIES) at spatial order FACE_K. Frames
without a student face box give a zero image (head down / away is itself a signal).
Temporal axis: Tchebichef order SEQ_K exactly as gtao_grid (pad_time + transform_1d).

Also stored, for the identity self-test only: the full-frame Tchebichef face computed with
gtao_grid's own transform_2d (must equal the paper/rich features where no padding occurs).

Interface = extract_features.py (init_worker, process), so the v5 Kaggle driver runs it.
Per zip: OUT/<stem>.npz with keys, subjects, n_frames, face_frac, feat_face_full,
feat_crop_<family> (layout (values, seq) raveled, values = FACE_K x FACE_K) + <stem>.json.
"""
import json
import shutil
import time
import traceback
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image

import moment_families as M
from _common import atomic_savez, load_grid

G = None
HDD_LOCK = None
MIN_BOX = 40          # px; smaller dict boxes are treated as no face
MARGIN = 1.10


def init_worker(lock, grid_path, seq_k=None, face_k=None):
    global G, HDD_LOCK
    HDD_LOCK = lock
    G = load_grid(grid_path)
    G.SEQ_K = int(seq_k or 8)
    G.FACE_K = int(face_k or 16)


def student_box(meta):
    box = ((meta.get("person") or {}).get("face") or {}).get("bounding_box")
    if not isinstance(box, dict):
        return None
    try:
        x0, y0, x1, y1 = (float(box[k]) for k in ("x0", "y0", "x1", "y1"))
    except (KeyError, TypeError, ValueError):
        return None
    if min(x1 - x0, y1 - y0) < MIN_BOX:
        return None
    return x0, y0, x1, y1


def crop_pixels(grey, box):
    x0, y0, x1, y1 = box
    cx, cy, half = (x0 + x1) / 2, (y0 + y1) / 2, max(x1 - x0, y1 - y0) * MARGIN / 2
    region = (int(round(cx - half)), int(round(cy - half)), int(round(cx + half)), int(round(cy + half)))
    crop = grey.crop(region).resize((G.FACE_SIZE, G.FACE_SIZE))   # outside the frame -> black
    return np.asarray(crop, dtype=np.float32) / 255.0


def encode(windows, subject_root):
    rows = []
    zeros = np.zeros((G.FACE_K, G.FACE_K), np.float32)
    for window in windows:
        full, crops, has_face = [], {f: [] for f in M.FAMILIES}, []
        for _, image_path in window["frames"]:
            # same three calls as gtao_grid.encode_windows -> identical full-frame face
            grey = Image.open(image_path).convert("L")
            pixels = np.asarray(grey.resize((G.FACE_SIZE, G.FACE_SIZE)), dtype=np.float32) / 255.0
            full.append(G.transform_2d(pixels, G.FACE_K))
            box = None
            meta_path = subject_root / "metadata" / f"{image_path.stem}.json"
            if meta_path.exists():
                try:
                    box = student_box(json.loads(meta_path.read_text()))
                except json.JSONDecodeError:
                    box = None
            has_face.append(box is not None)
            face = crop_pixels(grey, box) if box is not None else None
            for f in M.FAMILIES:
                crops[f].append(M.moments_2d(face, f, G.FACE_K) if face is not None else zeros)
        if not full:
            continue
        row = {"key": window["key"], "subject": window["subject"],
               "n_frames": len(full), "face_frac": float(np.mean(has_face)),
               "face_full": G.sequence_moments(G.pad_time(np.stack(full), G.SEQ_K), -1)}
        for f in M.FAMILIES:
            row[f"crop_{f}"] = G.sequence_moments(G.pad_time(np.stack(crops[f]), G.SEQ_K), -1)
        rows.append(row)
    return rows


def process(job):
    bundle_path, key, stem, out_dir, work_dir = job
    out_dir, work_dir = Path(out_dir), Path(work_dir)
    target = work_dir / stem
    start = time.time()
    try:
        shutil.rmtree(target, ignore_errors=True)
        target.mkdir(parents=True, exist_ok=True)
        with HDD_LOCK:
            with zipfile.ZipFile(bundle_path) as archive:
                archive.extractall(target)
        rows = []
        for subject_folder in sorted(p.parent for p in target.rglob("images") if p.is_dir()):
            windows = G.subject_windows(subject_folder, key)
            if windows:
                rows.extend(encode(windows, subject_folder))
        n, dim = len(rows), G.FACE_K * G.FACE_K * G.SEQ_K
        arrays = {"keys": np.asarray([r["key"] for r in rows]),
                  "subjects": np.asarray([r["subject"] for r in rows]),
                  "n_frames": np.asarray([r["n_frames"] for r in rows], dtype=np.int64),
                  "face_frac": np.asarray([r["face_frac"] for r in rows], dtype=np.float32)}
        for name in ["face_full"] + [f"crop_{f}" for f in M.FAMILIES]:
            arrays[f"feat_{name}"] = (np.stack([r[name] for r in rows]).astype(np.float32)
                                      if n else np.zeros((0, dim), np.float32))
        atomic_savez(out_dir / f"{stem}.npz", **arrays)
        summary = {"bundle": str(bundle_path), "key": key, "windows": n,
                   "face_frac_mean": float(arrays["face_frac"].mean()) if n else None,
                   "total_s": round(time.time() - start, 1), "grid_sha256": G.__grid_sha256__,
                   "seq_k": G.SEQ_K, "face_k": G.FACE_K, "families": list(M.FAMILIES)}
        (out_dir / f"{stem}.json").write_text(json.dumps(summary, indent=1))
        return stem, "ok", summary
    except Exception as error:
        return stem, "error", f"{type(error).__name__}: {error}\n{traceback.format_exc()}"
    finally:
        shutil.rmtree(target, ignore_errors=True)


if __name__ == "__main__":       # local test: python extract_families.py <zip> <key> <out>
    import multiprocessing as mp
    import sys
    zip_path, key, out = sys.argv[1:4]
    init_worker(mp.Lock(), None)
    stem = key.replace("/", "__")
    print(process((zip_path, key, stem, out, out + "/work"))[1:])
