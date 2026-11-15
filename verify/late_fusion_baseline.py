"""DIPSER decision-level (late) fusion baseline, computed from the released predictions.

Run from the repository root:

    python verify/late_fusion_baseline.py

For every seed, the four single-stream models (face, face mesh, head pose,
physiology) each predict a class for every window, out of fold. Late fusion
combines these four predictions per window, without any training:
  * plurality vote (ties go to the lower class, like the hard labels);
  * ordinal median (lower median of the four predicted levels).
It is done once with the CORAL single streams (unimodal_<s>_cw) and once with
the soft-target single streams (unimodal_<s>_cw_soft). The result is compared
with the best single stream (face mesh, soft target).
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "verify"))
from verify_results import qwk  # noqa: E402

RAW = ROOT / "results" / "v2" / "raw_dipser" / "gtao_grid_v1"
STREAMS = ("face", "facemesh", "head_pose", "physio")
SEEDS = (0, 1, 2)


def load_predictions(config, seed):
    """{window key: (true class, predicted class)} for one run."""
    oof = json.loads((RAW / config / ("seed" + str(seed)) / "oof.json").read_text())["oof"]
    result = {}
    for key, y, pred in zip(oof["keys"], oof["y"], oof["pred"]):
        result[key] = (int(y), int(pred))
    return result


def plurality(values):
    counts = np.bincount(np.asarray(values), minlength=3)
    return int(counts.argmax())


def lower_median(values):
    ordered = sorted(values)
    return int(ordered[(len(ordered) - 1) // 2])


def main():
    print("DIPSER late fusion of the four single-stream predictions (QWK, mean +- s.d. over 3 seeds)\n")
    for suffix, label in (("cw", "CORAL single streams"), ("cw_soft", "soft-target single streams")):
        vote_scores = []
        median_scores = []
        for seed in SEEDS:
            runs = []
            for stream in STREAMS:
                runs.append(load_predictions("unimodal_" + stream + "_" + suffix, seed))
            keys = sorted(runs[0])
            truth = []
            vote = []
            median = []
            for key in keys:
                predictions = []
                for run in runs:
                    predictions.append(run[key][1])
                truth.append(runs[0][key][0])
                vote.append(plurality(predictions))
                median.append(lower_median(predictions))
            vote_scores.append(qwk(truth, vote))
            median_scores.append(qwk(truth, median))
        print("  %-27s plurality vote %.3f +- %.3f   ordinal median %.3f +- %.3f" % (
            label, np.mean(vote_scores), np.std(vote_scores), np.mean(median_scores), np.std(median_scores)))
    best = []
    for seed in SEEDS:
        run = load_predictions("unimodal_facemesh_cw_soft", seed)
        truth = []
        pred = []
        for key in sorted(run):
            truth.append(run[key][0])
            pred.append(run[key][1])
        best.append(qwk(truth, pred))
    print("\n  best single stream (face mesh, soft):  %.3f +- %.3f" % (np.mean(best), np.std(best)))


if __name__ == "__main__":
    main()
