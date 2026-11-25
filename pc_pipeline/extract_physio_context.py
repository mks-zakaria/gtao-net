"""Physio-only re-extraction for Phase 3b fix 2: 4 / 16 / 32 s trailing physio windows.

For every window already in the per-zip feature files (extract_features.py output),
compute the physio vector (same 8 channels, same SEQ_K Tchebichef transform, same
"every channel needs >= 1 sample, edge-pad to SEQ_K" rule as gtao_grid.encode_windows)
over a trailing window [t_end - L, t_end) for L = 4, 16 and 32 s.

The 4 s level is the self-test: it must reproduce feat_physio / avail_physio of the
existing per-zip .npz exactly. Every zip's .json reports the check; --merge refuses to
write physio_context.npz if any zip failed it.

Only the watch-sensor members of each zip are unpacked (no images): the frame times
needed to rebuild each window's exact float t_start come from the zip's file listing.

Usage (PC):
  python extract_physio_context.py --data D:\\DIPSER --features C:\\dipser_features ^
      --out C:\\dipser_features\\physio_ctx --work C:\\dipser_work --workers 4
  python extract_physio_context.py --merge --out C:\\dipser_features\\physio_ctx ^
      --features C:\\dipser_features
-> C:\\dipser_features\\physio_ctx\\physio_context.npz  (upload next to corpus_full.npz)
"""
import argparse
import json
import multiprocessing as mp
import shutil
import time
import traceback
import zipfile
from pathlib import Path, PurePosixPath

import numpy as np

from _common import atomic_savez, find_bundles, load_grid

LENGTHS = (4, 16, 32)
WATCH_NAMES = ("watch_sensors", "watch_sensor")
G = None
HDD_LOCK = None


def init_worker(lock, grid_path):
    global G, HDD_LOCK
    G = load_grid(grid_path)
    HDD_LOCK = lock


def physio_vector(physio, t_start, t_end):
    """gtao_grid.encode_windows' physio rule on [t_start, t_end); None if missing."""
    parts = []
    for channel_name in G.PHYSIO_CHANNELS:
        channel = physio.get(channel_name)
        if channel is None:
            return None
        times, values = channel
        x = values[(times >= t_start) & (times < t_end)]
        if len(x) == 0:
            return None
        if len(x) < G.SEQ_K:
            x = np.pad(x, (0, G.SEQ_K - len(x)), mode="edge")
        parts.append(G.transform_1d(x[None, :], G.SEQ_K).ravel())
    return np.concatenate(parts)


def window_starts(image_names):
    """{'%.2f' % t: t} for the window grid of one subject folder (as subject_windows)."""
    times = []
    for name in sorted(image_names):
        try:
            times.append(G.parse_ts(PurePosixPath(name).stem))
        except (ValueError, IndexError):
            continue
    if not times:
        return {}
    first, last = float(times[0]), float(times[-1])
    starts, t = {}, first
    while t + G.WINDOW_S <= last + 1e-9:
        starts.setdefault(f"{t:.2f}", t)
        t += G.WINDOW_S
    return starts


def process(job):
    bundle_path, key, stem, feat_dir, out_dir, work_dir = job
    feat_dir, out_dir, work_dir = Path(feat_dir), Path(out_dir), Path(work_dir)
    target = work_dir / f"physio__{stem}"
    start = time.time()
    try:
        base = np.load(feat_dir / f"{stem}.npz", allow_pickle=False)
        keys = base["keys"].astype(str)
        shutil.rmtree(target, ignore_errors=True)
        target.mkdir(parents=True, exist_ok=True)
        with HDD_LOCK:
            with zipfile.ZipFile(bundle_path) as archive:
                names = archive.namelist()
                watch = [n for n in names
                         if any(f"/{w}/" in f"/{n}" for w in WATCH_NAMES) and not n.endswith("/")]
                for member in watch:
                    archive.extract(member, target)
        # subject folders = parents of an images/ folder (same rule as extract_features)
        folders = {}
        for name in names:
            parts = PurePosixPath(name).parts
            if "images" in parts and name.lower().endswith(".png"):
                prefix = PurePosixPath(*parts[:parts.index("images")])
                folders.setdefault(prefix, []).append(name)
        exact_t = {}
        physio_of = {}
        for prefix in sorted(folders):
            starts = window_starts(folders[prefix])
            physio = {}
            for w in WATCH_NAMES:
                if (target / prefix / w).is_dir():
                    physio = G.read_watch_sensors(target / prefix / w)
                    break
            for k, t in starts.items():
                if k not in exact_t:
                    exact_t[k] = t
                    physio_of[k] = physio
        n = len(keys)
        dim = len(G.PHYSIO_CHANNELS) * G.SEQ_K
        arrays = {"keys": keys}
        for length in LENGTHS:
            arrays[f"feat_physio_{length}"] = np.full((n, dim), np.nan, dtype=np.float32)
            arrays[f"avail_physio_{length}"] = np.zeros(n, dtype=bool)
        unmatched = 0
        for i, full_key in enumerate(keys):
            stamp = full_key.rsplit("#", 1)[1]
            if stamp not in exact_t:
                unmatched += 1
                continue
            t_end = exact_t[stamp] + G.WINDOW_S
            for length in LENGTHS:
                t_from = exact_t[stamp] if length == G.WINDOW_S else t_end - length
                vector = physio_vector(physio_of[stamp], t_from, t_end)
                if vector is not None:
                    arrays[f"feat_physio_{length}"][i] = vector
                    arrays[f"avail_physio_{length}"][i] = True
        # self-test: the 4 s level must equal the existing features
        same_avail = bool(np.array_equal(arrays["avail_physio_4"], base["avail_physio"]))
        a = base["avail_physio"]
        max_diff = (float(np.abs(arrays["feat_physio_4"][a] - base["feat_physio"][a]).max())
                    if a.any() and same_avail else (0.0 if same_avail else None))
        ok = same_avail and unmatched == 0 and (max_diff is not None and max_diff <= 1e-5)
        atomic_savez(out_dir / f"{stem}.npz", **arrays)
        summary = {"stem": stem, "windows": int(n), "unmatched_keys": unmatched,
                   "avail": {L: int(arrays[f"avail_physio_{L}"].sum()) for L in LENGTHS},
                   "selftest_avail_equal": same_avail, "selftest_max_abs_diff": max_diff,
                   "selftest_ok": ok, "watch_members": len(watch),
                   "total_s": round(time.time() - start, 1),
                   "grid_sha256": G.__grid_sha256__}
        (out_dir / f"{stem}.json").write_text(json.dumps(summary, indent=1))
        return stem, "ok", summary
    except Exception as error:
        return stem, "error", f"{type(error).__name__}: {error}\n{traceback.format_exc()}"
    finally:
        shutil.rmtree(target, ignore_errors=True)


def merge(out_dir, feat_dir):
    """Concatenate per-zip files in corpus_full.npz key order -> physio_context.npz."""
    out_dir, feat_dir = Path(out_dir), Path(feat_dir)
    corpus_keys = np.load(feat_dir / "corpus_full.npz", allow_pickle=False)["keys"].astype(str)
    files = sorted(out_dir.glob("group_*.npz"))
    failed = []
    for f in files:
        info = json.loads(f.with_suffix(".json").read_text())
        if not info["selftest_ok"]:
            failed.append(f.stem)
    if failed:
        raise SystemExit(f"self-test failed for {len(failed)} zips, e.g. {failed[:5]}: not merging")
    parts = [np.load(f, allow_pickle=False) for f in files]
    keys = np.concatenate([p["keys"].astype(str) for p in parts])
    position = {k: i for i, k in enumerate(keys)}
    missing = [k for k in corpus_keys if k not in position]
    if missing:
        raise SystemExit(f"{len(missing)} corpus keys have no physio context, e.g. {missing[:3]}")
    order = np.array([position[k] for k in corpus_keys])
    arrays = {"keys": corpus_keys}
    for length in LENGTHS:
        arrays[f"feat_physio_{length}"] = np.concatenate(
            [p[f"feat_physio_{length}"] for p in parts])[order]
        arrays[f"avail_physio_{length}"] = np.concatenate(
            [p[f"avail_physio_{length}"] for p in parts])[order]
    atomic_savez(out_dir / "physio_context.npz", **arrays)
    report = {"zips": len(files), "windows": int(len(corpus_keys)),
              "avail_pct": {L: round(float(arrays[f"avail_physio_{L}"].mean()), 4)
                            for L in LENGTHS}}
    (out_dir / "physio_context.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", help="folder with the downloaded zips (HDD)")
    parser.add_argument("--features", required=True,
                        help="extract_features.py output folder (per-zip .npz + corpus_full.npz)")
    parser.add_argument("--out", required=True, help="output folder for the physio context")
    parser.add_argument("--work", help="temporary unzip folder (SSD)")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--grid", default=None, help="path to gtao_grid.py")
    parser.add_argument("--merge", action="store_true", help="only merge finished per-zip files")
    args = parser.parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.merge:
        merge(out_dir, args.features)
        return
    work_dir = Path(args.work)
    work_dir.mkdir(parents=True, exist_ok=True)
    grid = load_grid(args.grid)
    bundles = find_bundles(args.data)
    todo = [(str(p), k, s, args.features, str(out_dir), str(work_dir))
            for p, k, s in bundles
            if (Path(args.features) / f"{s}.npz").exists() and not (out_dir / f"{s}.npz").exists()]
    print(f"{len(bundles)} zips, {len(todo)} to process with {args.workers} workers "
          f"(feature code sha256 {grid.__grid_sha256__[:12]})")
    start, done, bad = time.time(), 0, []
    lock = mp.Manager().Lock()
    with mp.Pool(args.workers, initializer=init_worker,
                 initargs=(lock, grid.__grid_path__)) as pool:
        for stem, status, info in pool.imap_unordered(process, todo):
            done += 1
            if status == "ok" and info["selftest_ok"]:
                print(f"[{done}/{len(todo)}] {stem}: {info['windows']} windows "
                      f"avail={info['avail']} ({info['total_s']}s)", flush=True)
            else:
                bad.append(stem)
                detail = info if status == "error" else json.dumps(info)
                print(f"[{done}/{len(todo)}] {stem}: PROBLEM {str(detail).splitlines()[0]}",
                      flush=True)
                with open(out_dir / "failed.log", "a") as log:
                    log.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {stem}\n{detail}\n")
    print(f"finished in {(time.time() - start) / 3600:.2f} h; {len(bad)} problems "
          f"(failed.log). Then run with --merge.")


if __name__ == "__main__":
    main()
