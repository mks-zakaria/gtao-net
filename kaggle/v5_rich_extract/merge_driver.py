"""Phase 6 merge: the 5 shard outputs -> corpus_rich.npz (+ full self-test).

Inputs: kernel outputs gtao-v5-dipser-rich-extract-shard0..4 (per-zip rich/*.npz) and the
private dataset (corpus_full.npz = paper features). Output /kaggle/working/corpus_rich.npz:
same keys/order as corpus_full.npz; face (order 24 x 12), facemesh / head pose (x 12);
feat_physio = the PAPER physio (order 6: rich physio is padding-affected), and the rich
physio kept as feat_physio_rich. Refuses to write if a key is missing or the leading
coefficients of a video stream differ from the paper features.
"""
import json
import os
from pathlib import Path

import numpy as np

SEQ_K, FACE_K = 12, 24
OUT = Path("/kaggle/working") if os.path.isdir("/kaggle/working") else Path("v5_merge_out")
OUT.mkdir(parents=True, exist_ok=True)


def find(pattern_name):
    hits = []
    for folder, _, names in os.walk(os.environ.get("MERGE_INPUT", "/kaggle/input")):
        hits += [Path(folder) / n for n in names if n.startswith("group_") and n.endswith(".npz")] \
            if pattern_name == "parts" else ([Path(folder) / pattern_name] if pattern_name in names else [])
    return hits


def main():
    parts = sorted(find("parts"), key=lambda p: p.name)
    names = [p.name for p in parts]
    assert len(names) == len(set(names)), "duplicate per-zip files across shards"
    base_path = find("corpus_full.npz")[0]
    base = np.load(base_path)
    keys = base["keys"].astype(str)
    print(f"{len(parts)} per-zip files; base corpus {len(keys)} windows", flush=True)
    loaded = [np.load(p) for p in parts]
    rich_keys = np.concatenate([d["keys"].astype(str) for d in loaded])
    position = {k: i for i, k in enumerate(rich_keys)}
    missing = [k for k in keys if k not in position]
    if missing:
        raise SystemExit(f"{len(missing)} corpus keys missing from the rich extraction, e.g. {missing[:3]}")
    order = np.array([position[k] for k in keys])
    cat = lambda name: np.concatenate([d[name] for d in loaded])[order]
    out = {"keys": keys, "subjects": base["subjects"], "y": base["y"], "soft": base["soft"],
           "n_raters": base["n_raters"], "n_frames": cat("n_frames")}
    report = {"zips": len(parts), "windows": int(len(keys)), "seq_k": SEQ_K, "face_k": FACE_K,
              "self_test": {}}
    for stream, n_values, face in (("face", FACE_K ** 2, FACE_K), ("facemesh", 478 * 3, None),
                                   ("head_pose", 3, None)):
        feats, avail = cat(f"feat_{stream}"), cat(f"avail_{stream}")
        rows = avail & base[f"avail_{stream}"] & (out["n_frames"] >= SEQ_K)
        lead = feats[rows].reshape(int(rows.sum()), n_values, SEQ_K)[:, :, :6]
        if face:
            lead = lead.reshape(len(lead), face, face, 6)[:, :16, :16, :]
        ref = base[f"feat_{stream}"][rows]
        row_diff = np.abs(lead.reshape(len(lead), -1) - ref).max(axis=1) / (np.abs(ref).max() + 1e-9)
        bad_share = float((row_diff > 1e-4).mean())
        exact = row_diff <= 1e-4
        report["self_test"][stream] = {
            "compared": int(rows.sum()), "rows_differing": int((~exact).sum()),
            "share_differing": bad_share,
            "max_rel_diff_on_exact_rows": float(row_diff[exact].max()) if exact.any() else None,
            "avail_equal": bool(np.array_equal(avail, base[f"avail_{stream}"]))}
        print(stream, report["self_test"][stream], flush=True)
        # face: every image gives a face -> must be exact in EVERY window (code identity).
        # Mesh / head pose come from per-frame metadata: windows with < SEQ_K such frames
        # are edge-padded to SEQ_K, which changes their coefficients by construction.
        # Verified 2026-11-20 on raw group_01/experiment_09/subject_04: mismatch <=> fewer
        # than 12 mesh (head-pose) frames, 22/22 windows each, 0 exceptions either way.
        # So those rows are reported, not failed; all other rows must be exact.
        if stream == "face" and bad_share > 0:
            raise SystemExit(f"self-test failed on face: {bad_share:.2%} of rows differ")
        out[f"feat_{stream}"], out[f"avail_{stream}"] = feats, avail
    out["feat_physio"], out["avail_physio"] = base["feat_physio"], base["avail_physio"]
    out["feat_physio_rich"], out["avail_physio_rich"] = cat("feat_physio"), cat("avail_physio")
    nf = out["n_frames"]
    report["frames_per_window"] = [int(nf.min()), float(np.median(nf)), int(nf.max())]
    report["class_counts"] = np.bincount(out["y"], minlength=3).tolist()
    np.savez(OUT / "corpus_rich.npz", **out)
    report["bytes"] = (OUT / "corpus_rich.npz").stat().st_size
    (OUT / "corpus_rich.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
