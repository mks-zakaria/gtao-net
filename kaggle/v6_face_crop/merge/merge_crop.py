"""Phase 7 merge: the 5 face-crop shard outputs -> corpus_crop.npz (+ identity self-test).

Inputs: kernel outputs gtao-v6-dipser-face-crop-shard0..4 (per-zip crop/*.npz) and the
private dataset (corpus_full.npz = paper features). Output corpus_crop.npz in corpus_full
key order: n_frames, face_frac (share of frames with the student's face box) and
feat_crop_<family> (face crop, spatial order 16 x temporal order 8, 5 families).
Self-test: the full-frame Tchebichef face computed by the same extractor must equal the
paper face features (leading 16x16x6) in every window with >= 8 frames (no padding at
either order). Refuses to write on a missing key or any mismatch.
"""
import json
import os
from pathlib import Path

import numpy as np

SEQ_K, FACE_K = 8, 16
FAMILIES = ("tcheb", "hahn", "kraw", "kraw_eyes", "kraw_mouth")
ROOT = os.environ.get("MERGE_INPUT", "/kaggle/input")
OUT = Path("/kaggle/working") if os.path.isdir("/kaggle/working") else Path("v6_merge_out")
OUT.mkdir(parents=True, exist_ok=True)


def main():
    parts, base_path = [], None
    for folder, _, names in os.walk(ROOT):
        for n in names:
            p = Path(folder) / n
            if n.startswith("group_") and n.endswith(".npz") and p.parent.name == "crop":
                parts.append(p)
            if n == "corpus_full.npz":
                base_path = p
    parts.sort(key=lambda p: p.name)
    names = [p.name for p in parts]
    assert len(names) == len(set(names)), "duplicate per-zip files across shards"
    base = np.load(base_path)
    keys = base["keys"].astype(str)
    print(f"{len(parts)} per-zip files; base corpus {len(keys)} windows", flush=True)
    loaded = [np.load(p) for p in parts]
    crop_keys = np.concatenate([d["keys"].astype(str) for d in loaded])
    position = {k: i for i, k in enumerate(crop_keys)}
    missing = [k for k in keys if k not in position]
    if missing:
        raise SystemExit(f"{len(missing)} corpus keys missing from the crop extraction, e.g. {missing[:3]}")
    order = np.array([position[k] for k in keys])
    cat = lambda name: np.concatenate([d[name] for d in loaded])[order]
    out = {"keys": keys, "n_frames": cat("n_frames"), "face_frac": cat("face_frac")}

    full = cat("feat_face_full")
    rows = base["avail_face"].astype(bool) & (out["n_frames"] >= SEQ_K)
    lead = full[rows].reshape(int(rows.sum()), FACE_K * FACE_K, SEQ_K)[:, :, :6].reshape(int(rows.sum()), -1)
    ref = base["feat_face"][rows]
    row_diff = np.abs(lead - ref).max(axis=1) / (np.abs(ref).max() + 1e-9)
    report = {"zips": len(parts), "windows": int(len(keys)), "seq_k": SEQ_K, "face_k": FACE_K,
              "families": list(FAMILIES),
              "self_test_full_frame_face": {"compared": int(rows.sum()),
                                            "rows_differing": int((row_diff > 1e-4).sum()),
                                            "max_rel_diff": float(row_diff.max())}}
    print(report["self_test_full_frame_face"], flush=True)
    if (row_diff > 1e-4).any() or rows.sum() == 0:
        raise SystemExit("self-test failed: full-frame face differs from the paper features")
    del full
    for f in FAMILIES:
        x = cat(f"feat_crop_{f}")
        assert np.isfinite(x).all(), f
        out[f"feat_crop_{f}"] = x
    ff = out["face_frac"]
    report["face_frac"] = {"mean": float(ff.mean()), "share_zero": float((ff == 0).mean()),
                           "share_one": float((ff == 1).mean()),
                           "mean_by_class": [float(ff[base["y"] == c].mean()) if (base["y"] == c).any() else None
                                             for c in range(3)]}
    np.savez(OUT / "corpus_crop.npz", **out)
    report["bytes"] = (OUT / "corpus_crop.npz").stat().st_size
    (OUT / "corpus_crop.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
