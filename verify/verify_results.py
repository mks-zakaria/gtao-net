"""Independent check of every gTAO-Net v2 result reported in the paper.

Run from the repository root:

    python verify/verify_results.py

It needs only numpy (scipy too for step 6, which reruns analysis/*.py). It does
NOT reuse the training code or the analysis scripts for steps 1-5: QWK, accuracy
and MAE are recomputed here from the raw out-of-fold predictions.

Steps
  1. Every oof.json: recompute QWK / accuracy / MAE / n / n_subjects from the
     raw labels and predictions and compare with the "pooled" block.
  2. Every row of every metrics JSON: the per-seed QWKs, the mean and the
     standard deviation must equal the values recomputed from the raw files.
  3. Subject-disjoint protocol: rebuild the outer folds with the kernels' rule
     (sorted subjects, numpy default_rng(seed).shuffle, round-robin into 5 folds),
     check the fold sizes against the recorded n_val, and check that no
     inner-validation subject (used for epoch selection) is in the scored fold.
  4. Permuted-label controls stay at |QWK| <= 0.10.
  5. Kaggle logs: the nested "inner split:" line is present and the shuffle
     control printed PASS.
  6. Rerun analysis/stats.py, tables.py, perclass.py, comparison.py on a
     temporary copy and compare their output with analysis/*.json and tables/.
  7. Headline numbers quoted in the abstract (verify/claims.json).

Exit code 0 = every check passed. Any failure is printed with "FAIL".
"""
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "v2"
N_FOLDS = 5
SEEDS = (0, 1, 2)
SHUFFLE_LIMIT = 0.10

FAILURES = []
COUNTS = {}


def fail(message):
    """Record one failed check and print it."""
    FAILURES.append(message)
    print("  FAIL " + message)


def count(name):
    """Count one passed check under `name`."""
    if name not in COUNTS:
        COUNTS[name] = 0
    COUNTS[name] = COUNTS[name] + 1


# ---------------------------------------------------------------------------
# Metrics, written from scratch (same definition as sklearn's
# cohen_kappa_score(weights="quadratic"), which the kernels call)
# ---------------------------------------------------------------------------
def qwk(y_true, y_pred):
    """Quadratic weighted kappa. Returns 0.0 where the kernels return 0.0
    (fewer than 2 samples, a single true class, or an undefined value)."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if len(y_true) < 2:
        return 0.0
    if len(np.unique(y_true)) < 2:
        return 0.0
    labels = sorted(set(y_true.tolist()) | set(y_pred.tolist()))
    index_of = {}
    for position, label in enumerate(labels):
        index_of[label] = position
    k = len(labels)
    observed = np.zeros((k, k))
    for a, b in zip(y_true.tolist(), y_pred.tolist()):
        observed[index_of[a], index_of[b]] += 1
    row_totals = observed.sum(axis=1)
    col_totals = observed.sum(axis=0)
    expected = np.outer(row_totals, col_totals) / observed.sum()
    weights = np.zeros((k, k))
    for i in range(k):
        for j in range(k):
            weights[i, j] = (i - j) ** 2
    denominator = float((weights * expected).sum())
    if denominator == 0.0:
        return 0.0
    value = 1.0 - float((weights * observed).sum()) / denominator
    if not math.isfinite(value):
        return 0.0
    return value


def close(a, b, tolerance):
    """True when two numbers agree within `tolerance`."""
    return abs(float(a) - float(b)) <= tolerance


# Values in the JSON files are rounded to 4 decimals.
TOL = 0.5e-4 + 1e-9


# ---------------------------------------------------------------------------
# Where the raw predictions of each metrics block live
# ---------------------------------------------------------------------------
def result_blocks():
    """List of (label, metrics rows, folder with <config>/seed<k>/oof.json, k-fold?, fold universe)."""
    blocks = []
    dipser = json.loads((RESULTS / "DIPSER_grid_v1_metrics.json").read_text())
    blocks.append(("DIPSER", dipser["summary"]["rows"],
                   RESULTS / "raw_dipser" / "gtao_grid_v1", True, read_dipser_subjects()))
    daisee = json.loads((RESULTS / "DAiSEE_metrics.json").read_text())
    base = RESULTS / "raw_daisee" / "daisee_gt_grid"
    blocks.append(("DAiSEE P1 (Validation)", daisee["protocol_P1_official"]["rows"],
                   base / "P1_official", False, None))
    blocks.append(("DAiSEE P2", daisee["protocol_P2_groupkfold"]["rows"],
                   base / "P2_groupkfold", True, None))
    blocks.append(("DAiSEE P1-Test", daisee["protocol_P1_test"]["rows"],
                   base / "P1_test", False, None))
    engagenet = json.loads((RESULTS / "EngageNet_metrics.json").read_text())
    blocks.append(("EngageNet P2", engagenet["protocol_P2_groupkfold"]["rows"],
                   RESULTS / "raw_engagenet" / "engagenet_gt_grid" / "P2_groupkfold", True, None))
    return blocks


# ---------------------------------------------------------------------------
# Step 1 and 3: one oof.json
# ---------------------------------------------------------------------------
def read_dipser_subjects():
    """The 40 DIPSER session-subjects the kernel deals into folds (see the file header)."""
    subjects = []
    for line in (ROOT / "verify" / "dipser_session_subjects.txt").read_text().splitlines():
        line = line.strip()
        if line != "" and not line.startswith("#"):
            subjects.append(line)
    return subjects


def outer_fold_of_subject(subjects, seed):
    """Rebuild the kernels' outer folds: subject -> fold number.

    `subjects` must be the kernel's full subject list. For DAiSEE and EngageNet
    every subject is scored, so the scored subjects are that list. For DIPSER the
    kernel deals all 40 session-subjects, then scores only complete windows."""
    unique_subjects = sorted(set(subjects))
    rng = np.random.default_rng(seed)
    rng.shuffle(unique_subjects)
    fold_of = {}
    for position, subject in enumerate(unique_subjects):
        fold_of[subject] = position % N_FOLDS
    return fold_of


def as_subject_list(text_or_list):
    """inner_val_subjects is a list, or (EngageNet/DAiSEE) a list stored as text."""
    if isinstance(text_or_list, list):
        return text_or_list
    cleaned = str(text_or_list).strip().lstrip("[").rstrip("]")
    subjects = []
    for part in cleaned.split(","):
        part = part.strip().strip("'").strip('"')
        if part != "":
            subjects.append(part)
    return subjects


def check_oof(label, path, seed, kfold, fold_universe):
    """Steps 1 and 3 for one oof.json. Returns the recomputed QWK.
    fold_universe: the kernel's full subject list (None = the scored subjects)."""
    doc = json.loads(path.read_text())
    oof = doc["oof"]
    y = np.asarray(oof["y"])
    pred = np.asarray(oof["pred"])
    subjects = []
    for subject in oof["subject"]:
        subjects.append(str(subject))
    name = label + " " + path.parent.parent.name + " seed" + str(seed)

    # Step 1: pooled metrics
    if len(y) != len(pred) or len(y) != len(subjects):
        fail(name + ": y, pred and subject have different lengths")
    if len(set(oof["keys"])) != len(oof["keys"]):
        fail(name + ": duplicate sample keys")
    value = qwk(y, pred)
    pooled = doc["pooled"]
    if not close(value, pooled["qwk"], TOL):
        fail(name + ": QWK %.6f != pooled %.4f" % (value, pooled["qwk"]))
    accuracy = float(np.mean(y == pred))
    if not close(accuracy, pooled["acc"], TOL):
        fail(name + ": accuracy %.6f != pooled %.4f" % (accuracy, pooled["acc"]))
    mae = float(np.mean(np.abs(y - pred)))
    if not close(mae, pooled["mae"], TOL):
        fail(name + ": MAE %.6f != pooled %.4f" % (mae, pooled["mae"]))
    if len(y) != int(pooled["n"]):
        fail(name + ": n %d != pooled %s" % (len(y), pooled["n"]))
    if len(set(subjects)) != int(pooled["n_subjects"]):
        fail(name + ": n_subjects differ")
    count("oof files: pooled QWK/acc/MAE/n recomputed")

    # Step 3: subject-disjoint outer folds, inner split inside the training fold
    folds = doc["folds"]
    scored_subjects = set(subjects)
    if kfold:
        if len(folds) != N_FOLDS:
            fail(name + ": %d folds, expected %d" % (len(folds), N_FOLDS))
        if fold_universe is None:
            fold_of = outer_fold_of_subject(subjects, seed)
        else:
            fold_of = outer_fold_of_subject(fold_universe, seed)
        sizes = [0] * N_FOLDS
        for subject in subjects:
            sizes[fold_of[subject]] += 1
        for record in folds:
            fold_number = int(record["fold"])
            if int(record["n_val"]) != sizes[fold_number]:
                fail(name + " fold %d: rebuilt size %d != recorded n_val %s"
                     % (fold_number, sizes[fold_number], record["n_val"]))
            if int(record["n_train"]) + int(record["n_val"]) != len(y):
                fail(name + " fold %d: n_train + n_val != n" % fold_number)
            inner = as_subject_list(record.get("inner_val_subjects", []))
            if len(inner) == 0:
                fail(name + " fold %d: no inner validation subjects (not nested)" % fold_number)
            for subject in inner:
                if subject not in fold_of:
                    fail(name + " fold %d: inner subject %s unknown" % (fold_number, subject))
                elif fold_of[subject] == fold_number:
                    fail(name + " fold %d: inner subject %s is in the scored fold"
                         % (fold_number, subject))
        count("k-fold runs: folds rebuilt, inner split outside scored fold")
    else:
        if len(folds) != 1:
            fail(name + ": %d folds, expected 1" % len(folds))
        record = folds[0]
        if int(record["n_val"]) != len(y):
            fail(name + ": n_val %s != scored n %d" % (record["n_val"], len(y)))
        inner = as_subject_list(record.get("inner_val_subjects", []))
        if len(inner) == 0:
            fail(name + ": no inner validation subjects (not nested)")
        for subject in inner:
            if subject in scored_subjects:
                fail(name + ": inner subject %s is also scored" % subject)
        count("official-split runs: inner split disjoint from scored subjects")
    return value


# ---------------------------------------------------------------------------
# Step 2 and 4: metrics rows
# ---------------------------------------------------------------------------
def check_block(label, rows, folder, kfold, fold_universe):
    """Steps 1-4 for one metrics block."""
    print("[%s] %d rows, raw predictions in %s" % (label, len(rows), folder.relative_to(ROOT)))
    on_disk = set()
    for child in folder.iterdir():
        if child.is_dir():
            on_disk.add(child.name)
    in_json = set()
    for row in rows:
        in_json.add(row["config"])
    for missing in sorted(in_json - on_disk):
        fail(label + ": no raw predictions for " + missing)
    for extra in sorted(on_disk - in_json):
        fail(label + ": raw predictions for %s but no metrics row" % extra)

    for row in rows:
        config = row["config"]
        if config not in on_disk:
            continue
        recomputed = []
        for seed in SEEDS:
            path = folder / config / ("seed" + str(seed)) / "oof.json"
            if not path.exists():
                fail(label + " " + config + ": missing " + path.name + " for seed " + str(seed))
                continue
            recomputed.append(check_oof(label, path, seed, kfold, fold_universe))
        reported = row["qwk_per_seed"]
        if len(recomputed) != len(reported):
            fail(label + " " + config + ": seed count differs")
            continue
        for seed_value, reported_value in zip(recomputed, reported):
            if not close(seed_value, reported_value, TOL):
                fail(label + " " + config + ": seed QWK %.6f != reported %.4f"
                     % (seed_value, reported_value))
        if not close(np.mean(recomputed), row["qwk_mean"], TOL + 1e-4):
            fail(label + " " + config + ": mean QWK differs")
        if not close(np.std(recomputed), row["qwk_std"], TOL + 1e-4):
            fail(label + " " + config + ": QWK std differs")
        count("metrics rows: per-seed, mean and std QWK recomputed")

        # Step 4: permuted-label control
        if row.get("shuffle"):
            worst = 0.0
            for value in recomputed:
                worst = max(worst, abs(value))
            print("  permuted-label control %s: max |QWK| = %.4f" % (config, worst))
            if worst > SHUFFLE_LIMIT:
                fail(label + " " + config + ": permuted control above %.2f" % SHUFFLE_LIMIT)
            count("permuted-label controls <= 0.10")


# ---------------------------------------------------------------------------
# Step 5: Kaggle logs
# ---------------------------------------------------------------------------
def check_logs():
    """The nested split ran and the shuffle control passed in every log."""
    logs = sorted(RESULTS.glob("*.log"))
    if len(logs) == 0:
        fail("no Kaggle logs in results/v2")
    for log in logs:
        text = log.read_text(errors="replace")
        inner_lines = text.count("inner split:")
        passed = "shuffle control" in text and "PASS" in text
        print("[log] %s: %d 'inner split' lines, shuffle control %s"
              % (log.name, inner_lines, "PASS" if passed else "missing"))
        if inner_lines == 0:
            fail(log.name + ": no 'inner split:' line (nested selection did not run)")
        if not passed:
            fail(log.name + ": shuffle control PASS line missing")
        count("logs checked")


# ---------------------------------------------------------------------------
# Step 6: rerun the analysis scripts and compare with the committed outputs
# ---------------------------------------------------------------------------
def check_analysis():
    """Regenerate stats.json, perclass.json and every tables/*.tex in a temp copy."""
    try:
        import scipy  # noqa: F401
    except ImportError:
        fail("scipy is not installed: cannot rerun analysis/*.py (pip install -r requirements.txt)")
        return
    work = Path(tempfile.mkdtemp(prefix="gtao_verify_"))
    shutil.copytree(ROOT / "analysis", work / "analysis")
    (work / "tables").mkdir()
    environment = dict(os.environ)
    environment["GTAO_RESULTS"] = str(RESULTS)
    for script in ("stats.py", "tables.py", "perclass.py", "comparison.py"):
        done = subprocess.run([sys.executable, str(work / "analysis" / script)],
                              cwd=str(work), env=environment,
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if done.returncode != 0:
            fail("analysis/%s failed:\n%s" % (script, done.stderr[-2000:]))
            return
    for name in ("stats.json", "perclass.json"):
        fresh = json.loads((work / "analysis" / name).read_text())
        committed = json.loads((ROOT / "analysis" / name).read_text())
        if fresh != committed:
            fail("analysis/%s regenerated from results/v2 differs from the committed file" % name)
        else:
            count("analysis JSON regenerated identically")
    for table in sorted((ROOT / "tables").glob("*.tex")):
        fresh_path = work / "tables" / table.name
        if not fresh_path.exists():
            fail("tables/%s is not produced by the analysis scripts" % table.name)
            continue
        if fresh_path.read_text() != table.read_text():
            fail("tables/%s regenerated from results/v2 differs from the paper's table" % table.name)
        else:
            count("paper tables regenerated identically")
    for fresh_path in sorted((work / "tables").glob("*.tex")):
        if not (ROOT / "tables" / fresh_path.name).exists():
            fail("analysis wrote tables/%s, which is not committed" % fresh_path.name)
    shutil.rmtree(work)


# ---------------------------------------------------------------------------
# Step 7: headline numbers of the abstract
# ---------------------------------------------------------------------------
def best_rows(rows):
    """(best single-stream row, best fusion row), shuffle controls excluded."""
    best_uni = None
    best_fusion = None
    for row in rows:
        if row.get("shuffle"):
            continue
        if row["fusion"] == "unimodal" or row.get("unimodal"):
            if best_uni is None or row["qwk_mean"] > best_uni["qwk_mean"]:
                best_uni = row
        else:
            if best_fusion is None or row["qwk_mean"] > best_fusion["qwk_mean"]:
                best_fusion = row
    return best_uni, best_fusion


def check_claims():
    """Compare the numbers quoted in the abstract with the results and stats.json."""
    claims = json.loads((ROOT / "verify" / "claims.json").read_text())
    stats = json.loads((ROOT / "analysis" / "stats.json").read_text())
    blocks = {}
    for label, rows, folder, kfold, fold_universe in result_blocks():
        blocks[label] = rows
    found = {}
    for label in ("DIPSER", "DAiSEE P1 (Validation)", "DAiSEE P2", "EngageNet P2"):
        best_uni, best_fusion = best_rows(blocks[label])
        found[label + " best single stream QWK"] = best_uni["qwk_mean"]
        found[label + " best single stream QWK sd"] = best_uni["qwk_std"]
        found[label + " best fusion QWK"] = best_fusion["qwk_mean"]
        found[label + " best fusion QWK sd"] = best_fusion["qwk_std"]
        found[label + " fusion minus best stream"] = round(
            best_fusion["qwk_mean"] - best_uni["qwk_mean"], 4)
        n_fusion = 0
        for row in blocks[label]:
            if row.get("shuffle"):
                continue
            if row["fusion"] == "unimodal" or row.get("unimodal"):
                continue
            n_fusion = n_fusion + 1
        found[label + " fusion configs"] = n_fusion
    perclass = json.loads((ROOT / "analysis" / "perclass.json").read_text())
    one_hot = 0
    gated_folds = 0
    for key in perclass:
        if key.startswith("dipser/") and perclass[key].get("gates"):
            one_hot = one_hot + perclass[key]["gates"]["one_hot"]
            gated_folds = gated_folds + perclass[key]["gates"]["n"]
    found["DIPSER gate one-hot folds"] = one_hot
    found["DIPSER gated folds"] = gated_folds
    found["DIPSER link wins"] = stats["dipser_link"]["wins"]
    found["DIPSER link pairs"] = stats["dipser_link"]["n_pairs"]
    found["DIPSER link p (config level)"] = stats["dipser_link"]["p_config"]
    found["DIPSER gate vs core p (config level)"] = stats["dipser_gate_vs_core"]["p_config"]
    found["DIPSER soft vs CORAL wins"] = stats["dipser_soft_vs_coral"]["wins"]
    found["DIPSER soft vs CORAL pairs"] = stats["dipser_soft_vs_coral"]["n_pairs"]
    found["DIPSER soft vs CORAL p (config level)"] = stats["dipser_soft_vs_coral"]["p_config"]
    found["DAiSEE P1 link wins"] = stats["daisee_P1_link"]["wins"]
    found["DAiSEE P2 link wins"] = stats["daisee_P2_link"]["wins"]
    test_uni, test_fusion = best_rows(blocks["DAiSEE P1-Test"])
    found["DAiSEE P1-Test best QWK (single stream)"] = test_uni["qwk_mean"]
    found["DAiSEE P1-Test accuracy % (single stream)"] = round(100 * test_uni["acc_mean"], 1)

    for key, claimed in claims.items():
        if key.startswith("_"):
            continue
        if key not in found:
            fail("claims.json key not computed: " + key)
            continue
        actual = found[key]
        if isinstance(claimed, int) and not isinstance(claimed, bool):
            ok = int(actual) == claimed
        else:
            ok = close(actual, claimed, 0.0006)
        status = "ok  "
        if not ok:
            status = "FAIL"
            FAILURES.append("claim %s: paper %s, results %s" % (key, claimed, actual))
        else:
            count("abstract claims match the results")
        print("  %s %-45s paper %-8s results %s" % (status, key, claimed, actual))


def main():
    print("gTAO-Net v2 verification, results in %s\n" % RESULTS.relative_to(ROOT))
    for label, rows, folder, kfold, fold_universe in result_blocks():
        check_block(label, rows, folder, kfold, fold_universe)
    print()
    check_logs()
    print("\n[analysis] rerunning analysis/*.py on results/v2")
    check_analysis()
    print("\n[claims] numbers quoted in the abstract")
    check_claims()
    print("\nSummary")
    for name in COUNTS:
        print("  %5d  %s" % (COUNTS[name], name))
    if len(FAILURES) > 0:
        print("\n%d check(s) FAILED" % len(FAILURES))
        sys.exit(1)
    print("\nALL CHECKS PASSED")


if __name__ == "__main__":
    main()
