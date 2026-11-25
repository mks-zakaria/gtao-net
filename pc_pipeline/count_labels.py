"""Count labelled 4-s windows per student and per experiment, straight from the zips.

Nothing is extracted: the script reads the image file names (they are the time
stamps) and the small rater JSON files from inside each zip, and applies the same
windowing, 5->3 collapse and plurality vote as gtao_grid.subject_windows.
Run it while the download is still going; it only looks at finished zips.

    python count_labels.py --data D:\\DIPSER --out C:\\dipser_features\\label_counts.csv

The key question it answers: how many HIGH-attention windows does each
experiment add? (The paper's slice has only 10, 7 of them in complete windows.)
"""
import argparse
import csv
import json
import time
import zipfile
from collections import defaultdict
from pathlib import Path, PurePosixPath

import numpy as np

from _common import find_bundles, load_grid


def read_raters_from_zip(archive, G):
    """Same parsing as gtao_grid.read_rater_labels, from zip members."""
    raters = {}
    names = sorted(n for n in archive.namelist()
                   if PurePosixPath(n).parent.name == "labels" and n.endswith(".json"))
    for name in names:
        path = PurePosixPath(name)
        if "__" in path.name:
            continue
        try:
            raw = json.loads(archive.read(name))
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue
        if not isinstance(raw, list):
            continue
        change_points = []
        for entry in raw:
            if not isinstance(entry, dict) or "datetime" not in entry:
                continue
            value = entry.get("attention")
            if value is None:
                continue
            change_points.append((G.parse_ts(entry["datetime"]), int(value)))
        raters[path.stem] = sorted(change_points, key=G.change_point_time)
    return raters


def count_bundle(bundle_path, G):
    """Return (class counts [low, med, high], n_windows, n_frames, window labels)."""
    with zipfile.ZipFile(bundle_path) as archive:
        frame_names = sorted(n for n in archive.namelist()
                             if PurePosixPath(n).parent.name == "images" and n.endswith(".png"))
        frame_times = []
        for name in frame_names:
            try:
                frame_times.append(G.parse_ts(PurePosixPath(name).stem))
            except (ValueError, IndexError):
                continue
        raters = read_raters_from_zip(archive, G)
    counts = np.zeros(G.N_CLASSES, dtype=int)
    labels = []
    if not frame_times or not raters:
        return counts, 0, len(frame_times), labels
    frame_times = np.array(frame_times, dtype=np.float64)
    t, last = float(frame_times[0]), float(frame_times[-1])
    while t + G.WINDOW_S <= last + 1e-9:
        middle = t + G.WINDOW_S / 2
        votes = [G.collapse_5_to_3(G.value_at(cp, middle))
                 for cp in raters.values() if G.value_at(cp, middle) is not None]
        if votes:
            low, high = np.searchsorted(frame_times, [t, t + G.WINDOW_S])
            if high > low:
                label = G.majority_vote(votes)
                counts[label] += 1
                labels.append(label)
        t += G.WINDOW_S
    return counts, int(counts.sum()), len(frame_times), labels


def between_share(labels_by_unit):
    """Share of label variance explained by unit (person or session) means."""
    y = np.concatenate([np.asarray(v, float) for v in labels_by_unit.values() if len(v)])
    if len(y) == 0 or y.var() == 0:
        return float("nan")
    means = np.concatenate([np.full(len(v), np.mean(v)) for v in labels_by_unit.values() if len(v)])
    return float(means.var() / y.var())


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", required=True, help="folder with the downloaded zips")
    parser.add_argument("--out", required=True, help="CSV file to write")
    parser.add_argument("--grid", default=None, help="path to gtao_grid.py")
    args = parser.parse_args()

    G = load_grid(args.grid)
    bundles = find_bundles(args.data)
    print(f"{len(bundles)} subject zips found under {args.data}")
    per_experiment = defaultdict(lambda: np.zeros(G.N_CLASSES, dtype=int))
    by_person = defaultdict(list)          # "group/subject_YY" -> labels (all sessions)
    by_session = defaultdict(list)
    rows = []
    start = time.time()
    for path, key, stem in bundles:
        try:
            counts, n_windows, n_frames, labels = count_bundle(path, G)
            status = "ok"
        except (zipfile.BadZipFile, OSError) as error:
            counts, n_windows, n_frames, labels = np.zeros(G.N_CLASSES, dtype=int), 0, 0, []
            status = f"error: {type(error).__name__}"
        group, experiment, subject = key.split("/")
        per_experiment[f"{group}/{experiment}"] += counts
        by_person[f"{group}/{subject}"].extend(labels)
        by_session[key].extend(labels)
        rows.append([group, experiment, subject, n_frames, n_windows, *counts.tolist(), status])
        print(f"  {key}: windows={n_windows} low/med/high={counts.tolist()} {status}", flush=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["group", "experiment", "subject", "frames", "windows",
                         "low", "medium", "high", "status"])
        writer.writerows(rows)

    print("\nPer experiment (low / medium / high):")
    total = np.zeros(G.N_CLASSES, dtype=int)
    for name in sorted(per_experiment):
        c = per_experiment[name]
        total += c
        print(f"  {name}: {c.tolist()}  windows={int(c.sum())}")
    print(f"TOTAL: {total.tolist()}  windows={int(total.sum())}  "
          f"high share={total[2] / max(total.sum(), 1):.2%}  ({time.time() - start:.0f}s)")
    print("\nPer group: persons (a person = group/subject_YY across sessions), sessions, "
          "and the share of label variance BETWEEN persons (trait-like labels):")
    for group in sorted({k.split("/")[0] for k in by_person}):
        persons = {k: v for k, v in by_person.items() if k.startswith(group + "/") and v}
        sessions = {k: v for k, v in by_session.items() if k.startswith(group + "/") and v}
        print(f"  {group}: {len(persons)} persons, {len(sessions)} sessions, "
              f"between-person share {between_share(persons):.0%}, "
              f"between-session share {between_share(sessions):.0%}")
    print(f"  ALL: {sum(1 for v in by_person.values() if v)} persons, "
          f"between-person share {between_share(by_person):.0%} "
          f"(paper slice: 59% between persons on complete windows)")
    print("Note: these are all labelled windows; the paper's complete-window population "
          "(all 4 streams present) is smaller.")


if __name__ == "__main__":
    main()
