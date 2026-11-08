"""Recompute every number in the gTAO-Net manuscript from gtao-net/results/*.json.
Run: python3 analysis/stats.py   (from dipseer/paper_gtao/)"""
import json, itertools
from pathlib import Path
import numpy as np
from scipy.stats import wilcoxon

import os
_here = Path(__file__).resolve()
_cands = [os.environ.get("GTAO_RESULTS", ""),                 # explicit override
          _here.parents[2] / "results" / "v1",                # portable bundle layout
          _here.parents[3] / "gtao-net" / "results"]          # phd repo layout
R = next(Path(c) for c in _cands if c and (Path(c) / "DIPSER_grid_v1_metrics.json").exists())
dip = json.load(open(R / "DIPSER_grid_v1_metrics.json"))
dai = json.load(open(R / "DAiSEE_metrics.json"))
# EngageNet may arrive after the other runs; its numbers then stay \pending in the paper
eng = json.load(open(R / "EngageNet_metrics.json")) if (R / "EngageNet_metrics.json").exists() else None
OUT = {}

def rows(lst): return {r["config"]: r for r in lst}
D = rows(dip["summary"]["rows"])

def name(f, s, d, l):
    n = f"{f}_skip{int(s)}_drop{int(d)}"
    return n if l == "coral" else f"{n}_{l}"

def paired(tab, pairs):
    diffs, wins = [], 0
    for a, b in pairs:
        da = np.array(tab[a]["qwk_per_seed"]) - np.array(tab[b]["qwk_per_seed"])
        diffs += da.tolist(); wins += tab[a]["qwk_mean"] > tab[b]["qwk_mean"]
    diffs = np.array(diffs)
    p = wilcoxon(diffs).pvalue if np.any(diffs) else 1.0
    cm = np.array([tab[a]["qwk_mean"] - tab[b]["qwk_mean"] for a, b in pairs])
    pc = wilcoxon(cm).pvalue if np.any(cm) else 1.0
    return dict(p_config=float(f"{pc:.2g}"),n_pairs=len(pairs), wins=int(wins), n_diffs=len(diffs),
                median=round(float(np.median(diffs)), 4), mean=round(float(diffs.mean()), 4),
                p=float(f"{p:.2g}"))

F, S, Dr, L = ("gate", "core"), (1, 0), (0, 1), ("coral", "hybrid", "soft")
OUT["dipser_link"] = paired(D, [(name(f,1,d,l), name(f,0,d,l)) for f in F for d in Dr for l in L])
OUT["dipser_gate_vs_core"] = paired(D, [(name("gate",s,d,l), name("core",s,d,l)) for s in S for d in Dr for l in L])
OUT["dipser_soft_vs_coral"] = paired(D, [(name(f,s,d,"soft"), name(f,s,d,"coral")) for f in F for s in S for d in Dr])
OUT["dipser_soft_vs_hybrid"] = paired(D, [(name(f,s,d,"soft"), name(f,s,d,"hybrid")) for f in F for s in S for d in Dr])
OUT["dipser_drop"] = paired(D, [(name(f,s,1,l), name(f,s,0,l)) for f in F for s in S for l in L])
# v2 adds soft-target unimodal baselines (unimodal_<s>_cw_soft); 4 pairs, so p_cfg >= 0.125
uni_soft = [(f"unimodal_{s}_cw_soft", f"unimodal_{s}_cw") for s in ("face", "facemesh", "head_pose", "physio")
            if f"unimodal_{s}_cw_soft" in D]
if uni_soft:
    OUT["dipser_unimodal_soft_vs_coral"] = paired(D, uni_soft)

# fusion vs best unimodal, seed by seed (same split per seed). "Best" is chosen from the
# data (v1: unimodal_facemesh_cw and gate_skip1_drop0_soft), so a re-run cannot silently
# compare the wrong pair.
fus = [r for r in D.values() if r["fusion"] in F and not r.get("shuffle")]
best_uni = max((r for r in D.values() if r["fusion"] == "unimodal" and not r.get("shuffle")), key=lambda r: r["qwk_mean"])
best_fus = max(fus, key=lambda r: r["qwk_mean"])
uni = best_uni["qwk_per_seed"]
OUT["dipser_best"] = [best_fus["config"], best_fus["qwk_mean"], best_uni["config"], best_uni["qwk_mean"],
                      round(best_fus["qwk_mean"] - best_uni["qwk_mean"], 4)]
OUT["dipser_fusion_vs_uni_bestpair_perseed"] = [round(a-b,4) for a,b in zip(best_fus["qwk_per_seed"], uni)]
OUT["dipser_fusion_seedcells_above_uni"] = [int(sum(r["qwk_per_seed"][i] > uni[i] for r in fus)) for i in range(len(uni))]
OUT["dipser_n_fusion_cfgs"] = len(fus)
OUT["dipser_shuffle"] = D.get("shuffle_core_skip1_drop1", {}).get("qwk_mean")
OUT["dipser_probes"] = {k: D[k]["probe_seed0"] for k in D if D[k].get("probe_seed0")}
pr = {k: round(v["tower_first_skipON"]/v["tower_first_skipOFF"],1) for k,v in OUT["dipser_probes"].items()}
OUT["dipser_probe_first_layer_ratio_range"] = [min(pr.values()), max(pr.values())]
OUT["dipser_probe_unique"] = sorted({(v["tower_first_skipON"], v["tower_first_skipOFF"]) for v in OUT["dipser_probes"].values()})

# masked eval
m = json.load(open(R / "DIPSER_h4_masked_eval.json"))
OUT["dipser_masked"] = {cfg: {k: round(float(np.mean(v["qwk"])) - float(np.mean(m[cfg]["none"]["qwk"])), 3)
                              for k, v in m[cfg].items()} for cfg in m}
OUT["dipser_masked_base"] = {cfg: round(float(np.mean(m[cfg]["none"]["qwk"])),4) for cfg in m}

# DAiSEE: soft == coral identity check, link / gate effects (coral+hybrid only)
for P in ("protocol_P1_official", "protocol_P2_groupkfold"):
    T = rows(dai[P]["rows"])
    ident = all(T[name(f,s,d,"soft")]["qwk_per_seed"] == T[name(f,s,d,"coral")]["qwk_per_seed"]
                for f in F for s in S for d in Dr)
    k = P.split("_")[1]
    OUT[f"daisee_{k}_soft_identical_to_coral"] = ident
    OUT[f"daisee_{k}_link"] = paired(T, [(name(f,1,d,l), name(f,0,d,l)) for f in F for d in Dr for l in ("coral","hybrid")])
    OUT[f"daisee_{k}_gate_vs_core"] = paired(T, [(name("gate",s,d,l), name("core",s,d,l)) for s in S for d in Dr for l in ("coral","hybrid")])
    fu = max((r for r in T.values() if r["fusion"] in F), key=lambda r: r["qwk_mean"])
    un = max((r for r in T.values() if r["fusion"] == "unimodal"), key=lambda r: r["qwk_mean"])
    OUT[f"daisee_{k}_best"] = [fu["config"], fu["qwk_mean"], un["config"], un["qwk_mean"], round(fu["qwk_mean"]-un["qwk_mean"],4)]
if eng is not None:
    E = rows(eng["protocol_P2_groupkfold"]["rows"])
    e_fu = max((r for r in E.values() if r["fusion"] in F and not r.get("shuffle")), key=lambda r: r["qwk_mean"])
    e_un = max((r for r in E.values() if r["fusion"] == "unimodal"), key=lambda r: r["qwk_mean"])
    OUT["engagenet_best"] = [e_fu["config"], e_fu["qwk_mean"], e_un["config"], e_un["qwk_mean"]]
    OUT["engagenet_gap"] = round(e_fu["qwk_mean"] - e_un["qwk_mean"], 4)
    OUT["engagenet_gate_vs_core_perseed"] = [round(a-b,4) for a,b in zip(E["gate_skip1_drop0"]["qwk_per_seed"], E["core_skip1_drop0"]["qwk_per_seed"])]
OUT["_results_dir"] = R.name   # tables.py refuses a stats.json computed from other results
print(json.dumps(OUT, indent=1, default=str))
json.dump(OUT, open(Path(__file__).with_name("stats.json"), "w"), indent=1, default=str)
