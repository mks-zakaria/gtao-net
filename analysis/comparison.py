"""Our rows of the DAiSEE comparison table (tab:comparison) and the numbers its prose quotes.

The published DAiSEE studies report four-class accuracy, a few also per-class recall, and none
reports QWK. This script writes:
  - a majority-class predictor on the same scored clips (its QWK is 0 by definition),
  - the best gTAO-Net single stream and the best gTAO-Net fusion model,
with accuracy, recall of the lowest engagement level and QWK. Class counts and recall come
from analysis/perclass.json (out-of-fold predictions); without it those cells show "--".

Which protocol: P1-Test when the results have it (the two P1 winners, chosen on Validation,
retrained on Train + Validation and scored on the official Test partition, like the published
studies); otherwise P1 (scored on Validation). The \\ifcmptest switch it writes tells the
manuscript which of the two it is.

Writes tables/comparison_ours.tex and tables/comparison_numbers.tex.
Run after perclass.py:  python3 analysis/comparison.py
"""
import json
import os
from pathlib import Path

H = Path(__file__).resolve().parent
T = H.parent / "tables"
_cands = [os.environ.get("GTAO_RESULTS", ""), H.parents[1] / "results" / "v1",
          H.parents[2] / "gtao-net" / "results"]
R = next(Path(c) for c in _cands if c and (Path(c) / "DIPSER_grid_v1_metrics.json").exists())

LOSS = {"coral": "CORAL", "hybrid": "hybrid", "soft": "soft"}
STREAM = {"face": "face", "facemesh": "face mesh", "head_pose": "head pose"}


def describe(row):
    """Readable name of a configuration, e.g. 'gate fusion, link off, dropout off, CORAL'."""
    if row["fusion"] == "unimodal":
        return f"{STREAM[row['unimodal']]} stream, {LOSS[row.get('loss') or 'coral']}"
    link = "on" if row["skip"] else "off"
    drop = "on" if row["mdrop"] else "off"
    return f"{row['fusion']} fusion, link {link}, dropout {drop}, {LOSS[row['loss']]}"


def best(rows, fusion_set):
    keep = [r for r in rows if r["fusion"] in fusion_set and not r.get("shuffle")]
    return max(keep, key=lambda r: r["qwk_mean"])


daisee = json.load(open(R / "DAiSEE_metrics.json"))
on_test = "protocol_P1_test" in daisee and len(daisee["protocol_P1_test"]["rows"]) == 2
if on_test:
    protocol, folder, set_name = "protocol_P1_test", "P1_test", "Test"
else:
    protocol, folder, set_name = "protocol_P1_official", "P1_official", "Val."
rows = daisee[protocol]["rows"]
uni = best(rows, {"unimodal"})
fus = best(rows, {"gate", "core"})
# The model quoted in the prose is also chosen on Validation (P1), never on Test.
p1_rows = daisee["protocol_P1_official"]["rows"]
if best(p1_rows, {"unimodal"})["qwk_mean"] > best(p1_rows, {"gate", "core"})["qwk_mean"]:
    overall = uni
else:
    overall = fus

perclass = {}
if (H / "perclass.json").exists():
    perclass = json.loads((H / "perclass.json").read_text())


def class_info(config):
    """(recall of level 0, true class counts) of a P1 configuration, or (None, None)."""
    entry = perclass.get(f"daisee/{folder}/{config}")
    if not entry:
        return None, None
    counts = [sum(r) // entry["n_seeds"] for r in entry["confusion_sum"]]
    return entry["recall_mean"][0], counts


def pct(x):
    return f"{100 * x:.1f}"


def thousands(n):
    """1429 -> 1{,}429 (LaTeX thousands separator)."""
    return f"{n:,}".replace(",", "{,}")


lines = []
short_lines = []  # main_short.tex: no Ref. column, our rows marked "(ours)"
n_scored = uni["n_median"]
_, counts = class_info(uni["config"])
if counts:
    majority_acc = max(counts) / sum(counts)
    majority_cell = pct(majority_acc)
else:
    majority_acc = None
    majority_cell = "--"
lines.append(f"Majority-class predictor & -- & {set_name}\\ ({thousands(n_scored)}) & {majority_cell} & 0.00 & 0.000 \\\\")
short_lines.append(f"Majority-class predictor & {set_name}\\ ({thousands(n_scored)}) & {majority_cell} & 0.00 & 0.000 \\\\")
for label, row in (("best single stream", uni), ("best fusion", fus)):
    recall0, _ = class_info(row["config"])
    rec = f"{recall0:.2f}" if recall0 is not None else "--"
    # Bold only the model chosen on Validation (the one the prose quotes), not the higher Test score.
    qwk = f"{row['qwk_mean']:.3f}"
    qwk = f"\\textbf{{{qwk}}}" if row is overall else qwk
    cells = (f"{set_name}\\ ({thousands(row['n_median'])}) & "
             f"{pct(row['acc_mean'])} & {rec} & {qwk} $\\pm$ {row['qwk_std']:.3f} \\\\")
    lines.append(f"gTAO-Net, {label} ({describe(row)}) & this work & {cells}")
    short_lines.append(f"gTAO-Net (ours), {label} ({describe(row)}) & {cells}")
(T / "comparison_ours.tex").write_text("\n".join(lines) + "\n")
(T / "comparison_ours_short.tex").write_text("\n".join(short_lines) + "\n")

macros = {
    "cmpBestName": describe(overall),
    "cmpBestQWK": f"{overall['qwk_mean']:.3f}",
    "cmpBestQWKsd": f"{overall['qwk_std']:.3f}",
    "cmpBestAcc": pct(overall["acc_mean"]),
    "cmpMajAcc": pct(majority_acc) if majority_acc is not None else "\\pending{majority-class accuracy}",
    "cmpNScored": thousands(n_scored),
}
text = "".join(f"\\newcommand{{\\{k}}}{{{v}}}\n" for k, v in macros.items())
text += "\\cmptesttrue\n" if on_test else "\\cmptestfalse\n"
(T / "comparison_numbers.tex").write_text(text)
print(f"comparison ({protocol}): best = {overall['config']} (QWK {overall['qwk_mean']:.3f}); "
      f"majority acc = {macros['cmpMajAcc']}")
