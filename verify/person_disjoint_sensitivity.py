"""DIPSER sensitivity analysis: rescore every run on person-disjoint windows only.

Run from the repository root:

    python verify/person_disjoint_sensitivity.py

Why: the DIPSER outer folds group windows by session-subject
(group/experiment/subject). The slice holds two sessions (experiment_01 and
experiment_05) of the same 20 students, so one student's other session can sit
in the training folds while the first is scored. Of the 23 session-subjects with
complete windows, 3 students appear in both sessions.

This script keeps every trained model and prediction as it is, and rescores each
run only on the scored windows whose student has NO session in the training
folds of that outer fold. It then repeats the paper's paired contrasts
(config-level Wilcoxon, as in analysis/stats.py) on that subset.
It is a post hoc check on a smaller subset, not a re-run with person-grouped folds.
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "verify"))
from verify_results import outer_fold_of_subject, qwk, read_dipser_subjects  # noqa: E402

RAW = ROOT / "results" / "v2" / "raw_dipser" / "gtao_grid_v1"
SEEDS = (0, 1, 2)
N_FOLDS = 5


def student_of(session_subject):
    """group_01/experiment_05/subject_12 -> group_01/subject_12."""
    parts = session_subject.split("/")
    return parts[0] + "/" + parts[2]


def score_run(path, fold_of):
    """(QWK on all windows, QWK on person-disjoint windows, n kept, n total)."""
    oof = json.loads(path.read_text())["oof"]
    subjects = oof["subject"]
    y = np.asarray(oof["y"])
    pred = np.asarray(oof["pred"])
    students_in_training = {}
    for fold_number in range(N_FOLDS):
        students = set()
        for subject in subjects:
            if fold_of[subject] != fold_number:
                students.add(student_of(subject))
        students_in_training[fold_number] = students
    keep_list = []
    for subject in subjects:
        seen = student_of(subject) in students_in_training[fold_of[subject]]
        keep_list.append(not seen)
    keep = np.array(keep_list)
    return qwk(y, pred), qwk(y[keep], pred[keep]), int(keep.sum()), len(y)


def config_name(fusion, skip, drop, loss):
    """Config folder name, e.g. gate_skip1_drop0_soft (CORAL has no suffix)."""
    name = fusion + "_skip" + str(skip) + "_drop" + str(drop)
    if loss != "coral":
        name = name + "_" + loss
    return name


def contrast(means, pairs):
    """Wins, median difference and config-level Wilcoxon p over (a, b) pairs."""
    differences = []
    wins = 0
    for a, b in pairs:
        difference = means[a] - means[b]
        differences.append(difference)
        if difference > 0:
            wins = wins + 1
    p_value = wilcoxon(np.array(differences)).pvalue
    return "%d/%d wins, median %+.3f, p_cfg = %.2g" % (
        wins, len(pairs), float(np.median(differences)), p_value)


def main():
    universe = read_dipser_subjects()
    fold_of_by_seed = {}
    for seed in SEEDS:
        fold_of_by_seed[seed] = outer_fold_of_subject(universe, seed)

    all_scores = {}
    clean_scores = {}
    kept_by_seed = {}
    for config_dir in sorted(RAW.iterdir()):
        if not config_dir.is_dir():
            continue
        all_scores[config_dir.name] = []
        clean_scores[config_dir.name] = []
        for seed in SEEDS:
            path = config_dir / ("seed" + str(seed)) / "oof.json"
            full, clean, kept, total = score_run(path, fold_of_by_seed[seed])
            all_scores[config_dir.name].append(full)
            clean_scores[config_dir.name].append(clean)
            kept_by_seed[seed] = (kept, total)

    print("Person-disjoint windows kept per seed (same subset for every config):")
    for seed in SEEDS:
        kept, total = kept_by_seed[seed]
        print("  seed %d: %d of %d windows (%.0f%%)" % (seed, kept, total, 100.0 * kept / total))

    for title, scores in (("All windows (as in the paper)", all_scores),
                          ("Person-disjoint windows only", clean_scores)):
        means = {}
        for name in scores:
            means[name] = float(np.mean(scores[name]))
        best_stream = None
        best_fusion = None
        for name in means:
            if name.startswith("unimodal"):
                if best_stream is None or means[name] > means[best_stream]:
                    best_stream = name
            elif name.startswith("gate") or name.startswith("core"):
                if best_fusion is None or means[name] > means[best_fusion]:
                    best_fusion = name
        above = 0
        n_fusion = 0
        for name in means:
            if name.startswith("gate") or name.startswith("core"):
                n_fusion = n_fusion + 1
                if means[name] > means[best_stream]:
                    above = above + 1
        print("\n" + title)
        print("  best single stream: %s  QWK %.3f +- %.3f" % (
            best_stream, means[best_stream], float(np.std(scores[best_stream]))))
        print("  best fusion:        %s  QWK %.3f +- %.3f" % (
            best_fusion, means[best_fusion], float(np.std(scores[best_fusion]))))
        print("  fusion configs above the best stream: %d / %d" % (above, n_fusion))
        print("  permuted-label control: %.3f" % means["shuffle_core_skip1_drop1"])

        link_pairs = []
        gate_pairs = []
        soft_pairs = []
        drop_pairs = []
        for fusion in ("gate", "core"):
            for drop in (0, 1):
                for loss in ("coral", "hybrid", "soft"):
                    link_pairs.append((config_name(fusion, 1, drop, loss),
                                       config_name(fusion, 0, drop, loss)))
            for skip in (0, 1):
                for drop in (0, 1):
                    soft_pairs.append((config_name(fusion, skip, drop, "soft"),
                                       config_name(fusion, skip, drop, "coral")))
                for loss in ("coral", "hybrid", "soft"):
                    drop_pairs.append((config_name(fusion, skip, 1, loss),
                                       config_name(fusion, skip, 0, loss)))
        for skip in (0, 1):
            for drop in (0, 1):
                for loss in ("coral", "hybrid", "soft"):
                    gate_pairs.append((config_name("gate", skip, drop, loss),
                                       config_name("core", skip, drop, loss)))
        print("  link on vs off:      " + contrast(means, link_pairs))
        print("  gate vs core:        " + contrast(means, gate_pairs))
        print("  soft vs CORAL:       " + contrast(means, soft_pairs))
        print("  dropout on vs off:   " + contrast(means, drop_pairs))


if __name__ == "__main__":
    main()
