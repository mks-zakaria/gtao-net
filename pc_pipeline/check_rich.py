"""Self-test of a rich extraction (--seq-k 12 --face-k 24) against the paper features.

    python check_rich.py --base C:\\dipser_features --rich C:\\dipser_features_rich

Tchebichef bases are orthogonal, so lower orders are leading coefficients: on windows
with >= 12 frames (same time basis in both runs) face[:16, :16, :6], facemesh / head
pose / physio [:, :6] of the rich run must equal the paper features. Exit 1 otherwise.
Also prints the frames-per-window distribution (decides whether order 12 is supported).
"""
import argparse
import sys
from pathlib import Path

import numpy as np

SEQ_BASE, FACE_BASE = 6, 16


def leading(x, n_values, seq_rich, face_rich=None):
    x = x.reshape(len(x), n_values, seq_rich)[:, :, :SEQ_BASE]
    if face_rich:
        x = x.reshape(len(x), face_rich, face_rich, SEQ_BASE)[:, :FACE_BASE, :FACE_BASE, :]
    return x.reshape(len(x), -1)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--base", required=True)
    ap.add_argument("--rich", required=True)
    ap.add_argument("--seq-k", type=int, default=12)
    ap.add_argument("--face-k", type=int, default=24)
    args = ap.parse_args()
    worst, checked, frames, compared = 0.0, 0, [], 0
    for rich_path in sorted(Path(args.rich).glob("group_*.npz")):
        base_path = Path(args.base) / rich_path.name
        if not base_path.exists():
            continue
        r, b = np.load(rich_path), np.load(base_path)
        assert np.array_equal(r["keys"], b["keys"]), rich_path.name
        frames.append(r["n_frames"])
        # same padding in both runs when the order is unchanged, or when no padding occurs
        ok = (r["n_frames"] >= args.seq_k) | (args.seq_k == SEQ_BASE)
        for stream, n_values, face in (("face", args.face_k ** 2, args.face_k),
                                       ("facemesh", 478 * 3, None), ("head_pose", 3, None)):
            # physio excluded: channels with < seq-k samples are edge-padded to seq-k, so a
            # higher order changes even the leading coefficients (by construction)
            rows = ok & r[f"avail_{stream}"] & b[f"avail_{stream}"]
            if not rows.any():
                continue
            compared += int(rows.sum())
            lead = leading(r[f"feat_{stream}"][rows], n_values, args.seq_k, face)
            diff = np.abs(lead - b[f"feat_{stream}"][rows]) / (np.abs(b[f"feat_{stream}"][rows]).max() + 1e-9)
            worst = max(worst, float(diff.max()))
        checked += 1
    frames = np.concatenate(frames) if frames else np.zeros(0)
    print(f"zips checked: {checked}; frames per window: min {frames.min() if len(frames) else '-'}, "
          f"median {np.median(frames) if len(frames) else '-'}, max {frames.max() if len(frames) else '-'}; "
          f"windows with >= {args.seq_k} frames: {(frames >= args.seq_k).mean():.1%}")
    print(f"windows compared: {compared}; max relative difference of leading "
          f"coefficients: {worst:.2e}")
    if compared == 0:
        print("RICH SELF-TEST NOT POSSIBLE: no window has >= seq-k frames "
              "(order seq-k is padding only: do not run the full rich extraction)")
        sys.exit(2)
    if worst > 1e-4:
        print("RICH SELF-TEST FAILED")
        sys.exit(1)
    print("RICH SELF-TEST PASSED")


if __name__ == "__main__":
    main()
