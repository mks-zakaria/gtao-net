"""Emit LaTeX tables from gtao-net/results/*.json and analysis/stats.json."""
import json
from pathlib import Path
H = Path(__file__).resolve().parent
import os
_here = Path(__file__).resolve()
_cands = [os.environ.get("GTAO_RESULTS", ""),                 # explicit override
          _here.parents[2] / "results" / "v1",                # portable bundle layout
          _here.parents[3] / "gtao-net" / "results"]          # phd repo layout
R = next(Path(c) for c in _cands if c and (Path(c) / "DIPSER_grid_v1_metrics.json").exists())
T = H.parent / "tables"
S = json.load(open(H / "stats.json"))
if S.get("_results_dir", R.name) != R.name:
    raise SystemExit(f"stats.json was computed from results/{S['_results_dir']}, not {R.name}: run stats.py first")
dip = {r["config"]: r for r in json.load(open(R / "DIPSER_grid_v1_metrics.json"))["summary"]["rows"]}
dai = json.load(open(R / "DAiSEE_metrics.json"))
eng = ({r["config"]: r for r in json.load(open(R / "EngageNet_metrics.json"))["protocol_P2_groupkfold"]["rows"]}
       if (R / "EngageNet_metrics.json").exists() else None)   # not yet run: row stays \pending
STREAM = {"face": "face", "facemesh": "face mesh", "head_pose": "head pose", "physio": "physiology"}
ORDER = ("face", "facemesh", "head_pose", "physio")

def f(x, d=3): return f"{x:+.{d}f}".replace("+", "") if x >= 0 else f"$-${abs(x):.{d}f}"
def pm(r): return f"{f(r['qwk_mean'])} $\\pm$ {r['qwk_std']:.3f}"

# ---- Table: full DIPSER grid -------------------------------------------------
lines = []
def cell(r):
    g = r.get("gate_alpha_final") or []
    a = f"{ORDER[max(range(4), key=lambda i: g[i])].replace('_',' ').replace('facemesh','face mesh')} ({max(g):.2f})" if g else "--"
    return a
uni = [k for k in dip if dip[k]["fusion"] == "unimodal"]
fus = [k for k in dip if dip[k]["fusion"] in ("gate", "core") and not dip[k].get("shuffle")]
for k in sorted(uni, key=lambda k: -dip[k]["qwk_mean"]):
    r = dip[k]
    obj = {"coral": "CORAL", "hybrid": "hybrid", "soft": "soft"}[r.get("loss") or "coral"]
    lines.append(f"{STREAM[r['unimodal']]} & -- & -- & {obj} & {pm(r)} & {r['acc_mean']:.3f} & {r['mae_mean']:.3f} & -- \\\\")
lines.append("\\midrule")
for k in sorted(fus, key=lambda k: -dip[k]["qwk_mean"]):
    r = dip[k]
    lines.append(f"{r['fusion']} & {'on' if r['skip'] else 'off'} & {'on' if r['mdrop'] else 'off'} & "
                 f"{ {'coral':'CORAL','hybrid':'hybrid','soft':'soft'}[r['loss']] } & {pm(r)} & {r['acc_mean']:.3f} & {r['mae_mean']:.3f} & {cell(r)} \\\\")
sh = [k for k in dip if dip[k].get("shuffle")][0]
r = dip[sh]
lines.append("\\midrule")
lines.append(f"\\multicolumn{{4}}{{l}}{{permuted labels (core, link, dropout, CORAL)}} & {pm(r)} & {r['acc_mean']:.3f} & {r['mae_mean']:.3f} & -- \\\\")
(T / "dipser_grid.tex").write_text("\n".join(lines) + "\n")

# ---- Table: paired factorial contrasts --------------------------------------
def pf(p):
    if p >= 0.01: return f"{p:.2f}"
    m, e = f"{p:.1e}".split("e"); return f"${m}\\times10^{{{int(e)}}}$"
def row(lbl, s, ds):
    return (f"{lbl} & {ds} & {s['wins']}/{s['n_pairs']} & {f(s['median'])} & {f(s['mean'])} & "
            f"{pf(s['p_config'])} & {pf(s['p'])} \\\\")
L = [row("Global link on vs.\\ off", S["dipser_link"], "DIPSER"),
     row("Gate vs.\\ core fusion", S["dipser_gate_vs_core"], "DIPSER"),
     row("Soft vs.\\ CORAL objective", S["dipser_soft_vs_coral"], "DIPSER"),
     row("Soft vs.\\ hybrid objective", S["dipser_soft_vs_hybrid"], "DIPSER"),
     row("Modality dropout on vs.\\ off", S["dipser_drop"], "DIPSER")]
if "dipser_unimodal_soft_vs_coral" in S:
    L.append(row("Soft vs.\\ CORAL, single stream", S["dipser_unimodal_soft_vs_coral"], "DIPSER"))
L += [
     "\\midrule",
     row("Global link on vs.\\ off", S["daisee_P1_link"], "DAiSEE P1"),
     row("Global link on vs.\\ off", S["daisee_P2_link"], "DAiSEE P2"),
     row("Gate vs.\\ core fusion", S["daisee_P1_gate_vs_core"], "DAiSEE P1"),
     row("Gate vs.\\ core fusion", S["daisee_P2_gate_vs_core"], "DAiSEE P2")]
(T / "contrasts.tex").write_text("\n".join(L) + "\n")

# ---- Table: masked inference --------------------------------------------------
M = S["dipser_masked"]; B = S["dipser_masked_base"]
L = []
for cfg, lab in (("gate_skip1_drop0_soft", "gate, link on, soft"), ("core_skip1_drop0_soft", "core, link on, soft")):
    L.append(f"{lab} & {B[cfg]:.3f} & " + " & ".join(f(M[cfg][f'mask_{s}']) for s in ORDER) + " \\\\")
(T / "masked.tex").write_text("\n".join(L) + "\n")

# ---- Table: transfer ----------------------------------------------------------
L = []
for P, lab in (("protocol_P1_official", "DAiSEE, P1 official split"), ("protocol_P2_groupkfold", "DAiSEE, P2 subject 5-fold")):
    rr = {r["config"]: r for r in dai[P]["rows"]}
    u = {s: rr[f"unimodal_{s}"] for s in ("face", "facemesh", "head_pose")}
    bf = max((r for r in rr.values() if r["fusion"] in ("gate", "core") and not r.get("shuffle")), key=lambda r: r["qwk_mean"])
    bu = max(u.values(), key=lambda r: r["qwk_mean"])
    shq = rr.get("shuffle_core_skip1_drop1")
    L.append(f"{lab} & {pm(u['face'])} & {pm(u['facemesh'])} & {pm(u['head_pose'])} & {pm(bf)} & {f(bf['qwk_mean']-bu['qwk_mean'])} & {f(shq['qwk_mean']) if shq else '--'} \\\\")
if eng is None:
    L.append("EngageNet, P2 subject 5-fold$^\\dagger$ & \\multicolumn{6}{c}{\\pending{EngageNet, from the v2 run}} \\\\")
else:
    u = {s: eng[f"unimodal_{s}"] for s in ("face", "facemesh", "head_pose")}
    bf = max((r for r in eng.values() if r["fusion"] in ("gate", "core") and not r.get("shuffle")), key=lambda r: r["qwk_mean"])
    bu = max(u.values(), key=lambda r: r["qwk_mean"])
    L.append(f"EngageNet, P2 subject 5-fold$^\\dagger$ & {pm(u['face'])} & {pm(u['facemesh'])} & {pm(u['head_pose'])} & {pm(bf)} & {f(bf['qwk_mean']-bu['qwk_mean'])} & {f(eng['shuffle_core_skip1_drop1']['qwk_mean'])} \\\\")
(T / "transfer.tex").write_text("\n".join(L) + "\n")

# ---- Protocol switch ----------------------------------------------------------
# The prose that describes early stopping differs between v1 (epoch chosen on the scored
# fold) and v2 (nested inner split). main.tex reads this flag; results/v2 or
# GTAO_PROTOCOL=nested turns the nested text on.
nested = R.name == "v2" or os.environ.get("GTAO_PROTOCOL") == "nested"
(T / "protocol.tex").write_text("\\nestedtrue\n" if nested else "\\nestedfalse\n")
print("best fusion per DAiSEE protocol:", [max((r for r in dai[P]['rows'] if r['fusion'] in ('gate','core')), key=lambda r: r['qwk_mean'])['config'] for P in ('protocol_P1_official','protocol_P2_groupkfold')])
