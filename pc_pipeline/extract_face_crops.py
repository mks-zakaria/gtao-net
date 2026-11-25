"""Phase 10B, step 1 (PC, CPU, HDD-bound): RGB face crops + per-rater votes for every window.

    python extract_face_crops.py --data D:\\DIPSER --out C:\\dipser_crops --work C:\\dipser_work --workers 5

Per subject zip it writes OUT/<stem>.npz (compressed) with, for each 4 s window of
gtao_grid.subject_windows (same windows and keys as corpus_full.npz):
  crops      uint8 [n, 8, 112, 112, 3]  up to 8 frames spread evenly over the window, cropped
                                        to the student's face box (extract_families rule: dict
                                        bounding box, square, 10% margin); black if no face
  valid      bool  [n, 8]               the slot holds a frame of the window
  has_face   bool  [n, 8]               that frame has a student face box
  votes      int8  [n, R]               each rater's 1..5 score at the window middle (0 = none)
  rater_names                           the R rater file stems of this subject
  keys, subjects, y                     as in corpus_full.npz (y = plurality label, 3 classes)
Step 2 is embed_faces.py (GPU). Resumable: finished zips are skipped.
"""
import argparse
import json
import multiprocessing as mp
import os
import shutil
import time
import traceback
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image

from _common import find_bundles, load_grid

G = None
HDD_LOCK = None
N_SLOTS = 8
SIZE = 112
MIN_BOX = 40          # px; same rule as extract_families.py
MARGIN = 1.10


def init_worker(lock, grid_path):
    global G, HDD_LOCK
    HDD_LOCK = lock
    G = load_grid(grid_path)


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


def crop_rgb(image, box):
    x0, y0, x1, y1 = box
    cx, cy, half = (x0 + x1) / 2, (y0 + y1) / 2, max(x1 - x0, y1 - y0) * MARGIN / 2
    region = (int(round(cx - half)), int(round(cy - half)), int(round(cx + half)), int(round(cy + half)))
    return np.asarray(image.crop(region).resize((SIZE, SIZE), Image.BILINEAR), dtype=np.uint8)


def pick(n):
    """Indices of up to N_SLOTS frames spread evenly over n frames."""
    if n <= N_SLOTS:
        return list(range(n))
    return sorted(set(int(round(v)) for v in np.linspace(0, n - 1, N_SLOTS)))


def encode(windows, subject_root, raters, names):
    n = len(windows)
    crops = np.zeros((n, N_SLOTS, SIZE, SIZE, 3), np.uint8)
    valid = np.zeros((n, N_SLOTS), bool)
    has_face = np.zeros((n, N_SLOTS), bool)
    votes = np.zeros((n, len(names)), np.int8)
    for w, window in enumerate(windows):
        frames = window["frames"]
        for slot, i in enumerate(pick(len(frames))):
            _, image_path = frames[i]
            valid[w, slot] = True
            box = None
            meta_path = subject_root / "metadata" / f"{image_path.stem}.json"
            if meta_path.exists():
                try:
                    box = student_box(json.loads(meta_path.read_text()))
                except json.JSONDecodeError:
                    box = None
            if box is not None:
                crops[w, slot] = crop_rgb(Image.open(image_path).convert("RGB"), box)
                has_face[w, slot] = True
        middle = (window["t_start"] + window["t_end"]) / 2
        for r, name in enumerate(names):
            value = G.value_at(raters[name], middle)
            votes[w, r] = 0 if value is None else int(value)
    return crops, valid, has_face, votes


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
        parts = []
        for subject_folder in sorted(p.parent for p in target.rglob("images") if p.is_dir()):
            windows = G.subject_windows(subject_folder, key)
            if not windows:
                continue
            raters = G.read_rater_labels(subject_folder / "labels")
            names = sorted(raters)
            parts.append((windows, names) + encode(windows, subject_folder, raters, names))
        names = sorted(set(n for p in parts for n in p[1]))
        keys, subjects, labels, crops, valid, has_face, votes = [], [], [], [], [], [], []
        for windows, part_names, c, v, h, vt in parts:
            keys += [w["key"] for w in windows]
            subjects += [w["subject"] for w in windows]
            labels += [w["label"] for w in windows]
            crops.append(c), valid.append(v), has_face.append(h)
            full = np.zeros((len(windows), len(names)), np.int8)
            for r, name in enumerate(part_names):
                full[:, names.index(name)] = vt[:, r]
            votes.append(full)
        n = len(keys)
        arrays = {"keys": np.asarray(keys), "subjects": np.asarray(subjects),
                  "y": np.asarray(labels, dtype=np.int64), "rater_names": np.asarray(names),
                  "crops": np.concatenate(crops) if n else np.zeros((0, N_SLOTS, SIZE, SIZE, 3), np.uint8),
                  "valid": np.concatenate(valid) if n else np.zeros((0, N_SLOTS), bool),
                  "has_face": np.concatenate(has_face) if n else np.zeros((0, N_SLOTS), bool),
                  "votes": np.concatenate(votes) if n else np.zeros((0, len(names)), np.int8)}
        tmp = out_dir / f"{stem}.npz.tmp"
        with open(tmp, "wb") as handle:
            np.savez_compressed(handle, **arrays)
        os.replace(tmp, out_dir / f"{stem}.npz")
        summary = {"bundle": str(bundle_path), "key": key, "windows": n, "raters": names,
                   "face_frac": float(arrays["has_face"].sum() / max(1, arrays["valid"].sum())),
                   "total_s": round(time.time() - start, 1), "grid_sha256": G.__grid_sha256__}
        (out_dir / f"{stem}.json").write_text(json.dumps(summary, indent=1))
        return stem, "ok", summary
    except Exception as error:
        return stem, "error", f"{type(error).__name__}: {error}\n{traceback.format_exc()}"
    finally:
        shutil.rmtree(target, ignore_errors=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", required=True, help="folder with the downloaded zips (HDD)")
    parser.add_argument("--out", required=True, help="crop folder (SSD, about 10 GB)")
    parser.add_argument("--work", required=True, help="temporary unzip folder (SSD)")
    parser.add_argument("--workers", type=int, default=5)
    parser.add_argument("--only", default=None, help="restrict to keys containing this text")
    parser.add_argument("--grid", default=None, help="path to gtao_grid.py")
    args = parser.parse_args()

    out_dir, work_dir = Path(args.out), Path(args.work)
    out_dir.mkdir(parents=True, exist_ok=True)
    work_dir.mkdir(parents=True, exist_ok=True)
    grid = load_grid(args.grid)
    bundles = find_bundles(args.data)
    if args.only:
        bundles = [b for b in bundles if args.only in b[1]]
    todo = [(str(p), k, s, str(out_dir), str(work_dir))
            for p, k, s in bundles if not (out_dir / f"{s}.npz").exists()]
    print(f"{len(bundles)} zips found, {len(bundles) - len(todo)} already done, "
          f"{len(todo)} to extract with {args.workers} workers", flush=True)
    if not todo:
        return
    start, done, failed = time.time(), 0, []
    lock = mp.Manager().Lock()
    with mp.Pool(args.workers, initializer=init_worker, initargs=(lock, grid.__grid_path__)) as pool:
        for stem, status, info in pool.imap_unordered(process, todo):
            done += 1
            eta_h = (time.time() - start) / done * (len(todo) - done) / 3600
            if status == "ok":
                print(f"[{done}/{len(todo)}] {stem}: {info['windows']} windows, face {info['face_frac']:.2f}, "
                      f"raters {len(info['raters'])} ({info['total_s']}s, ETA {eta_h:.1f} h)", flush=True)
            else:
                failed.append(stem)
                print(f"[{done}/{len(todo)}] {stem}: FAILED {info.splitlines()[0]}", flush=True)
                with open(out_dir / "failed.log", "a") as log:
                    log.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {stem}\n{info}\n")
    print(f"finished in {(time.time() - start) / 3600:.2f} h; {len(failed)} failed "
          f"(re-run to retry; details in failed.log)", flush=True)


if __name__ == "__main__":
    main()
