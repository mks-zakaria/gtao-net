"""Extract Tchebichef features from every downloaded DIPSER subject zip.

One output file per zip:  OUT/<group>__<experiment>__<subject>.npz  (+ .json summary).
It is resumable: zips whose .npz already exists are skipped, so re-run it any time
(for instance after more zips have finished downloading).

Disk layout on the PC (zips on the HDD, everything else on the SSD):

    python extract_features.py --data D:\\DIPSER --out C:\\dipser_features ^
                               --work C:\\dipser_work --workers 5

Only ONE worker reads from the HDD at a time (a lock around the unzip step), so the
hard disk reads sequentially instead of seeking between 5 files; the CPU-heavy part
(image decoding + Tchebichef moments) runs on all workers in parallel.

The features come from gtao_grid.py's own functions, so they are identical to the
paper's v2 features. Then run merge_features.py to build the corpus file.
"""
import argparse
import json
import multiprocessing as mp
import shutil
import time
import traceback
import zipfile
from pathlib import Path

import numpy as np

from _common import atomic_savez, feature_dims, find_bundles, load_grid

G = None
HDD_LOCK = None


def init_worker(lock, grid_path, seq_k=None, face_k=None):
    global G, HDD_LOCK
    HDD_LOCK = lock
    G = load_grid(grid_path)
    # optional richer moments (Phase 6): the grid functions read these globals at call time
    if seq_k:
        G.SEQ_K = int(seq_k)
    if face_k:
        G.FACE_K = int(face_k)


def watch_folder(subject_folder):
    """The grid reads 'watch_sensors'; DATASET.md documents 'watch_sensor'. Accept both."""
    for name in ("watch_sensors", "watch_sensor"):
        if (subject_folder / name).is_dir():
            return subject_folder / name, name
    return subject_folder / "watch_sensors", None


def process(job):
    bundle_path, key, stem, out_dir, work_dir = job
    out_dir, work_dir = Path(out_dir), Path(work_dir)
    target = work_dir / stem
    start = time.time()
    try:
        shutil.rmtree(target, ignore_errors=True)
        target.mkdir(parents=True, exist_ok=True)
        with HDD_LOCK:                                   # sequential HDD reads
            with zipfile.ZipFile(bundle_path) as archive:
                archive.extractall(target)
        unzip_s = time.time() - start

        subject_folders = sorted(p.parent for p in target.rglob("images") if p.is_dir())
        rows, watch_names, frames_of = [], [], {}
        for subject_folder in subject_folders:
            windows = G.subject_windows(subject_folder, key)
            folder, name = watch_folder(subject_folder)
            physio = G.read_watch_sensors(folder)
            watch_names.append(name)
            if windows:
                rows.extend(G.encode_windows(windows, subject_folder, physio))
                for w in windows:
                    frames_of[w["key"]] = len(w["frames"])

        dims = feature_dims(G)
        n = len(rows)
        arrays = {
            "keys": np.asarray([r["key"] for r in rows]),
            "subjects": np.asarray([r["subject"] for r in rows]),
            "y": np.asarray([r["label"] for r in rows], dtype=np.int64),
            "soft": (np.stack([r["soft"] for r in rows]).astype(np.float32)
                     if n else np.zeros((0, G.N_CLASSES), np.float32)),
            "n_raters": np.asarray([r["n_raters"] for r in rows], dtype=np.int64),
            "n_frames": np.asarray([frames_of.get(r["key"], 0) for r in rows], dtype=np.int64),
        }
        for stream in G.STREAMS:
            matrix = np.full((n, dims[stream]), np.nan, dtype=np.float32)
            avail = np.zeros(n, dtype=bool)
            for i, r in enumerate(rows):
                if r[stream] is not None:
                    matrix[i] = r[stream]
                    avail[i] = True
            arrays[f"feat_{stream}"] = matrix
            arrays[f"avail_{stream}"] = avail
        atomic_savez(out_dir / f"{stem}.npz", **arrays)

        summary = {
            "bundle": str(bundle_path), "key": key, "windows": n,
            "class_counts": np.bincount(arrays["y"], minlength=G.N_CLASSES).tolist(),
            "avail": {s: int(arrays[f"avail_{s}"].sum()) for s in G.STREAMS},
            "watch_folder": watch_names, "subject_folders": len(subject_folders),
            "unzip_s": round(unzip_s, 1), "total_s": round(time.time() - start, 1),
            "grid_sha256": G.__grid_sha256__,
            "seq_k": G.SEQ_K, "face_k": G.FACE_K,
            "frames_per_window": ([int(np.min(arrays["n_frames"])), float(np.median(arrays["n_frames"])),
                                   int(np.max(arrays["n_frames"]))] if n else None),
        }
        (out_dir / f"{stem}.json").write_text(json.dumps(summary, indent=1))
        return stem, "ok", summary
    except Exception as error:                            # keep going with other zips
        return stem, "error", f"{type(error).__name__}: {error}\n{traceback.format_exc()}"
    finally:
        shutil.rmtree(target, ignore_errors=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", required=True, help="folder with the downloaded zips (HDD)")
    parser.add_argument("--out", required=True, help="feature folder (SSD)")
    parser.add_argument("--work", required=True, help="temporary unzip folder (SSD)")
    parser.add_argument("--workers", type=int, default=5)
    parser.add_argument("--only", default=None,
                        help="restrict to keys containing this text, e.g. group_01/experiment_03")
    parser.add_argument("--grid", default=None, help="path to gtao_grid.py")
    parser.add_argument("--seq-k", type=int, default=None,
                        help="temporal Tchebichef order (paper: 6; Phase 6 rich: 12)")
    parser.add_argument("--face-k", type=int, default=None,
                        help="2-D face Tchebichef order (paper: 16; Phase 6 rich: 24)")
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
          f"{len(todo)} to extract with {args.workers} workers")
    print(f"feature code: {grid.__grid_path__} (sha256 {grid.__grid_sha256__[:12]})")
    if not todo:
        return

    start, done, failed = time.time(), 0, []
    lock = mp.Manager().Lock()
    with mp.Pool(args.workers, initializer=init_worker,
                 initargs=(lock, grid.__grid_path__, args.seq_k, args.face_k)) as pool:
        for stem, status, info in pool.imap_unordered(process, todo):
            done += 1
            elapsed = time.time() - start
            eta_h = elapsed / done * (len(todo) - done) / 3600
            if status == "ok":
                print(f"[{done}/{len(todo)}] {stem}: {info['windows']} windows "
                      f"{info['class_counts']} avail={info['avail']} "
                      f"({info['total_s']}s, ETA {eta_h:.1f} h)", flush=True)
                if info["watch_folder"] == [None]:
                    print(f"    ! no watch folder in {stem}: physio will be missing", flush=True)
            else:
                failed.append(stem)
                print(f"[{done}/{len(todo)}] {stem}: FAILED {info.splitlines()[0]}", flush=True)
                with open(out_dir / "failed.log", "a") as log:
                    log.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {stem}\n{info}\n")
    print(f"finished in {(time.time() - start) / 3600:.2f} h; {len(failed)} failed "
          f"(re-run to retry; details in failed.log)")


if __name__ == "__main__":
    main()
