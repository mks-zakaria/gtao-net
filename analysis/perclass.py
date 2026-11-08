"""Per-class recall and gate behaviour across all folds and seeds, from the raw oof.json
files that tools/pull_v2.sh keeps in results/<RES>/raw_<dataset>/.

Answers two review items that the pooled metrics cannot: (1) how each class is recognised,
in particular the DIPSER high class, and (2) whether the gate collapse seen at seed 0, fold 0
holds over every fold and seed.

Writes analysis/perclass.json, tables/perclass.tex and tables/gates.tex. When no raw files are
present (v1), it writes empty tables and says so; main.tex only shows them under \\ifnested.
Run after stats.py:  python3 analysis/perclass.py
"""
import json
import os
from collections import Counter
from pathlib import Path

import numpy as np

H = Path(__file__).resolve().parent
T = H.parent / "tables"
_cands = [os.environ.get("GTAO_RESULTS", ""), H.parents[1] / "results" / "v1",
          H.parents[2] / "gtao-net" / "results"]
R = next(Path(c) for c in _cands if c and (Path(c) / "DIPSER_grid_v1_metrics.json").exists())

DATASETS = {  # raw folder -> (label, class names, stream order of the gate vector)
    "dipser": ("DIPSER", ("low", "medium", "high"), ("face", "facemesh", "head_pose", "physio")),
    "daisee": ("DAiSEE", ("0", "1", "2", "3"), ("face", "facemesh", "head_pose")),
    "engagenet": ("EngageNet", ("not", "barely", "engaged", "highly"), ("face", "facemesh", "head_pose")),
}
STREAM_LABEL = {"face": "face", "facemesh": "face mesh", "head_pose": "head pose", "physio": "physiology"}
ONE_HOT = 0.98   # a gate vector is "one-hot" when one stream gets at least this weight


def load_runs(raw):
    """Return {(protocol, config): [run, ...]} with one run per seed."""
    runs = {}
    for f in sorted(raw.rglob("oof.json")):
        d = json.loads(f.read_text())
        if "oof" not in d:
            continue
        config = f.parent.parent.name
        protocol = f.parent.parent.parent.name     # e.g. P1/P2 folder; one value on DIPSER
        runs.setdefault((protocol, config), []).append(d)
    return runs


def recall(runs, n_classes):
    """Per-class recall for each seed, plus the confusion matrix summed over seeds."""
    per_seed, cm = [], np.zeros((n_classes, n_classes), dtype=int)
    for d in runs:
        y, p = np.array(d["oof"]["y"]), np.array(d["oof"]["pred"])
        m = np.zeros((n_classes, n_classes), dtype=int)
        np.add.at(m, (y, p), 1)
        cm += m
        support = m.sum(axis=1)
        per_seed.append([float(m[c, c] / support[c]) if support[c] else float("nan")
                         for c in range(n_classes)])
    return np.array(per_seed), cm


def gate_summary(runs, streams):
    """Final-epoch gate vector of every fold of every seed."""
    finals = []
    for d in runs:
        for trace in d.get("gate_trace", {}).values():
            if trace and trace[-1].get("gate"):
                finals.append(trace[-1]["gate"])
    if not finals:
        return None
    winners = Counter(streams[int(np.argmax(g))] for g in finals)
    return {"n": len(finals),
            "one_hot": sum(max(g) >= ONE_HOT for g in finals),
            "winners": dict(winners.most_common()),
            "mean_max_weight": round(float(np.mean([max(g) for g in finals])), 3)}


def best(rows, fusion_set):
    keep = [r for r in rows if r["fusion"] in fusion_set and not r.get("shuffle")]
    return max(keep, key=lambda r: r["qwk_mean"])["config"] if keep else None


summary_rows = {
    "dipser": {"": json.load(open(R / "DIPSER_grid_v1_metrics.json"))["summary"]["rows"]},
    "daisee": {k: v["rows"] for k, v in json.load(open(R / "DAiSEE_metrics.json")).items()
               if isinstance(v, dict) and "rows" in v},
}
if (R / "EngageNet_metrics.json").exists():
    summary_rows["engagenet"] = {k: v["rows"] for k, v in json.load(open(R / "EngageNet_metrics.json")).items()
                                 if isinstance(v, dict) and "rows" in v}

OUT, recall_lines, gate_lines = {}, [], []
for key, (label, classes, streams) in DATASETS.items():
    raw = R / f"raw_{key}"
    if not raw.is_dir():
        continue
    runs = load_runs(raw)
    for (protocol, config), seed_runs in sorted(runs.items()):
        per_seed, cm = recall(seed_runs, len(classes))
        gates = gate_summary(seed_runs, streams)
        OUT[f"{key}/{protocol}/{config}"] = {
            "n_seeds": len(seed_runs),
            "recall_mean": np.round(np.nanmean(per_seed, axis=0), 3).tolist(),
            "confusion_sum": cm.tolist(),
            "gates": gates}
    # Table rows: best fusion and best unimodal of each protocol
    # Run folders: DIPSER has one (gtao_grid_v1); DAiSEE/EngageNet use P1_official and
    # P2_groupkfold, matching the "protocol_<folder>" keys of the metrics JSON.
    for protocol_key, rows in summary_rows[key].items():
        folder = protocol_key.replace("protocol_", "")
        for kind, fset in (("best single stream", {"unimodal"}), ("best fusion", {"gate", "core"})):
            cfg = best(rows, fset)
            match = [v for (p, c), v in ((tuple(k.split("/")[1:]), v) for k, v in OUT.items()
                                         if k.startswith(f"{key}/"))
                     if c == cfg and (not folder or p == folder)]
            if not cfg or not match:
                continue
            # a class absent from the scored set has no recall: shown as "--"
            rec = " / ".join("--" if x != x else f"{x:.2f}" for x in match[0]["recall_mean"])
            # P1_official -> "P1", P2_groupkfold -> "P2", P1_test -> "P1-Test"
            short = {"P1_test": "P1-Test"}.get(folder, folder.split("_")[0])
            name = label if not folder else f"{label} {short}"
            tt = cfg.replace("_", "\\_")
            recall_lines.append(f"{name} & {kind} & \\texttt{{{tt}}} & {rec} \\\\")
    # Gate rows: every gated configuration, pooled over folds and seeds
    for k, v in OUT.items():
        if k.startswith(f"{key}/") and v["gates"] and "/gate_" in k:
            g = v["gates"]
            top = ", ".join(f"{STREAM_LABEL[s]} {n}" for s, n in g["winners"].items())
            protocol, cfg = k.split("/")[1], k.split("/")[-1].replace("_", "\\_")
            short = {"P1_test": "P1-Test"}.get(protocol, protocol.split("_")[0])
            name = label if key == "dipser" else f"{label} {short}"
            gate_lines.append(f"{name} &\\texttt{{{cfg}}} & {g['one_hot']}/{g['n']} & "
                              f"{g['mean_max_weight']:.2f} & {top} \\\\")

(H / "perclass.json").write_text(json.dumps(OUT, indent=1))
(T / "perclass.tex").write_text("\n".join(recall_lines) + "\n")
(T / "gates.tex").write_text("\n".join(gate_lines) + "\n")
if not OUT:
    print(f"no raw oof.json under {R}/raw_* — per-class and gate tables left empty")
else:
    print(f"per-class recall for {len(OUT)} runs; {len(gate_lines)} gated configurations")
