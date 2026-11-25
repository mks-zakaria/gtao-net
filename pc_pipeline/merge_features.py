"""Merge the per-zip feature files into ONE corpus file the grid can train on.

The output has exactly the format of gtao_grid.build_corpus's cache
(``corpus_grid_v1.npz``): put it in the grid's output folder and the grid loads it
instead of re-reading the zips.

    python merge_features.py --features C:\\dipser_features --out corpus_grid_v1.npz
    python merge_features.py --features C:\\dipser_features --out paper_slice.npz ^
        --only group_01/experiment_01 group_01/experiment_05      # the paper's 40 zips

Copy the result (~1-2 GB for the whole dataset) to the M5.
"""
import argparse
import json
from pathlib import Path

import numpy as np

STREAMS = ("face", "facemesh", "head_pose", "physio")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--features", required=True, help="folder written by extract_features.py")
    parser.add_argument("--out", required=True, help="corpus .npz to write")
    parser.add_argument("--only", nargs="*", default=None,
                        help="keep only keys starting with one of these, e.g. group_01/experiment_01")
    args = parser.parse_args()

    # per-zip files only: a merged corpus written to the same folder must not be re-merged
    files = sorted(Path(args.features).glob("group_*.npz"))  # sorted by stem = grid order
    parts = []
    for path in files:
        part = np.load(path, allow_pickle=False)
        if len(part["y"]) == 0:
            continue
        if args.only and not any(str(part["subjects"][0]).startswith(p) for p in args.only):
            continue
        parts.append(part)
    if not parts:
        raise SystemExit("no feature files matched")

    def cat(name):
        return np.concatenate([p[name] for p in parts])

    keys = np.asarray(cat("keys").tolist())
    subjects = np.asarray(cat("subjects").tolist())
    y = cat("y").astype(np.int64)
    out = {"keys": keys, "subjects": subjects, "y": y,
           "soft": cat("soft").astype(np.float32), "n_raters": cat("n_raters").astype(np.int64)}
    if all("n_frames" in p.files for p in parts):          # rich extraction (Phase 6)
        out["n_frames"] = cat("n_frames").astype(np.int64)
    for stream in STREAMS:
        avail = cat(f"avail_{stream}")
        out[f"avail_{stream}"] = avail
        # build_corpus stores a stream that is never present as an empty (0, 0) matrix
        out[f"feat_{stream}"] = (cat(f"feat_{stream}") if avail.any()
                                 else np.zeros((0, 0), dtype=np.float32))
    np.savez(args.out, **out)

    complete = np.ones(len(y), dtype=bool)
    for stream in STREAMS:
        complete &= out[f"avail_{stream}"]
    report = {
        "zips": len(parts), "windows": int(len(y)),
        "subjects": len(set(subjects.tolist())),
        "class_counts": np.bincount(y, minlength=3).tolist(),
        "complete_windows": int(complete.sum()),
        "complete_class_counts": np.bincount(y[complete], minlength=3).tolist(),
        "avail_pct": {s: round(float(out[f"avail_{s}"].mean()), 3) for s in STREAMS},
    }
    print(json.dumps(report, indent=1))
    Path(args.out).with_suffix(".json").write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
