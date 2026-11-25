"""Tables and prose numbers for the paper's full-corpus section (never hand-edit the outputs).

Reads only results of record (Kaggle kernels; statistics recomputed by gtao-v9-statistics):
  results/v9/results/v3b/gate3.json      Phase 3b (full-corpus grid, Gate 3)
  results/v9/results/logs/label_prior.txt label-only references
  results/v9/results/v4/split_leakage.json random vs person-grouped split
  results/v9/results/v4/rich_vs_paper.json, results/v4/*/final.json  hyper-parameter search
  results/v9/results/v6/phase7.json      face crop + moment families
  results/v9/results/v8/phase8.json, results/v8/summary.json    time context
  results/v9/results/v10/phase9.json, results/v10/summary.json  class context
  results/v11/phase10a.json             sparse-probe tracking (kernel gtao-v11-dipser-sparse-probe)
  results/v12_3enc/phase10b.json        frozen face encoders + self-report probe (kernel gtao-v12-dipser-face-encoder v2)
Writes paper/tables/fullcorpus_numbers.tex (macros), fullcorpus_tests.tex, fullcorpus_protocol.tex,
fullcorpus_probe.tex.
    python analysis_v3/paper_fullcorpus.py
"""
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
R, K = ROOT / "results", ROOT / "results" / "v9" / "results"
T = ROOT / "paper" / "tables"
load = lambda p: json.loads(Path(p).read_text())


def q(x):                       # QWK / r, signed only when negative
    return f"{x:.3f}" if x >= 0 else f"$-${abs(x):.3f}"


def d(x):                       # signed difference
    return f"$+${x:.3f}" if x >= 0 else f"$-${abs(x):.3f}"


def ci(c):
    return f"[{d(c[0])}, {d(c[1])}]"


def pct(x):
    return f"{100 * x:.1f}"


def p(x):
    return "$<$0.001" if x < 0.001 else f"{x:.2f}" if x >= 0.01 else f"{x:.3f}"


def main():
    gate = load(K / "v3b" / "gate3.json")
    leak = load(K / "v4" / "split_leakage.json")
    rich = load(K / "v4" / "rich_vs_paper.json")
    p7, p8, p9 = load(K / "v6" / "phase7.json"), load(K / "v8" / "phase8.json"), load(K / "v10" / "phase9.json")
    s8, s10 = load(R / "v8" / "summary.json"), load(R / "v10" / "summary.json")
    fin = {r: load(R / "v4" / r / "final.json") for r in ("dipser_acc", "dipser_rich_acc", "dipser_crop_acc")}
    prior = (K / "logs" / "label_prior.txt").read_text()
    share = dict(re.findall(r"(person|experiment\(group-exp\)|session) ([0-9.]+)", prior))
    exp_prior = float(re.search(r"experiment-prior \(leave-person-out\) QWK ([0-9.]+)", prior).group(1))
    moment_prior = float(re.search(r"class-moment prior .*? QWK ([0-9.]+)", prior).group(1))

    g, pt = gate["gate"], gate["point"]
    single, fusion = gate["best_single"], gate["best_fusion"]
    cm = gate["vs_classmate"]
    best_cm = min(cm, key=lambda s: cm[s]["diff_p"])
    rnd, pg = leak["random stratified (windows)"], leak["person-grouped stratified"]
    a10 = s10["arms"]
    fam_q = p7["family"]["qwk"]["hahn"]
    crop = p7["crop_vs_scene"]["acc"]
    macros = {
        "fcPersons": "57", "fcSessions": "483", "fcWindows": "35{,}649", "fcComplete": "28{,}926",
        "fcShareHigh": "15.1",
        "fcSharePerson": f"{100 * float(share['person']):.0f}",
        "fcShareClass": f"{100 * float(share['experiment(group-exp)']):.0f}",
        "fcShareSession": f"{100 * float(share['session']):.0f}",
        "fcExpPrior": f"{exp_prior:.3f}", "fcMomentPrior": f"{moment_prior:.3f}",
        "fcBestSingleQWK": q(pt[single]["qwk"]),
        "fcBestSingleCI": f"[{gate['ci'][single]['qwk'][0]:.3f}, {gate['ci'][single]['qwk'][1]:.3f}]",
        "fcBestFusionQWK": q(pt[fusion]["qwk"]),
        "fcFusionDiff": d(g["d_diff"]), "fcFusionDiffCI": ci(g["d_ci"]), "fcFusionP": p(g["d_p"]),
        "fcFusionsAbove": g["d_fusions_above_best_single"].replace("/", " of "),
        "fcWithinSessionMax": f"{max(pt[c]['within_session'] for c in gate['best_stream'].values()):.3f}",
        "fcPartialR": f"{g['c_best_single']['partial_r']:.3f}",
        "fcClassmateStream": best_cm.replace("_", " "), "fcClassmateP": p(cm[best_cm]["diff_p"]),
        "fcLeakRandAcc": pct(rnd["acc"][0]), "fcLeakRandQWK": q(rnd["qwk"][0]),
        "fcLeakPersonAcc": pct(pg["acc"][0]), "fcLeakPersonQWK": q(pg["qwk"][0]),
        "fcMajority": pct(rnd["majority_acc"]),
        "fcSearchPaperAcc": pct(fin["dipser_acc"]["test_mean_sd"]["acc"][0]),
        "fcSearchRichAcc": pct(fin["dipser_rich_acc"]["test_mean_sd"]["acc"][0]),
        "fcSearchCropAcc": pct(fin["dipser_crop_acc"]["test_mean_sd"]["acc"][0]),
        "fcSearchCropQWK": q(fin["dipser_crop_acc"]["test_mean_sd"]["qwk"][0]),
        "fcSearchMajority": pct(fin["dipser_acc"]["info"]["majority_test"]["acc"]),
        "fcCropAccGain": f"{100 * crop['diff_acc']:.1f}",
        "fcCropAccCI": f"[{100 * crop['ci_acc'][0]:+.1f}, {100 * crop['ci_acc'][1]:+.1f}]",
        "fcCtxNoneQWK": q(s8["arms"]["none"]["qwk"][0]), "fcCtxGruQWK": q(s8["arms"]["gru"]["qwk"][0]),
        "fcCtxBiQWK": q(s8["arms"]["bigru"]["qwk"][0]),
        "fcLogoOwnAcc": pct(a10["logo_own"]["acc"][0]), "fcLogoOwnQWK": q(a10["logo_own"]["qwk"][0]),
        "fcLogoClassQWK": q(a10["logo_own_class"]["qwk"][0]),
        "fcPgOwnQWK": q(a10["pg5_own"]["qwk"][0]), "fcPgClassQWK": q(a10["pg5_own_class"]["qwk"][0]),
        "fcLeakClassDiff": d(p9["exploratory_pg5_own_class_vs_own"]["diff_qwk"]),
        "fcLeakClassCI": ci(p9["exploratory_pg5_own_class_vs_own"]["ci_qwk"]),
        "fcLogoDrop": d(p9["exploratory_own_logo_vs_pg5"]["diff_qwk"]),
        "fcLogoDropCI": ci(p9["exploratory_own_logo_vs_pg5"]["ci_qwk"]),
        "fcLogoSeedZeroQWK": q(p9["exploratory_own_logo_vs_pg5"]["b_qwk"]),
    }
    p10 = load(R / "v11" / "phase10a.json")
    pts = lambda c: f"[$+${100 * c[0]:.1f}, $+${100 * c[1]:.1f}]"
    probe_lines = ["% written by analysis_v3/paper_fullcorpus.py -- do not edit\n"]
    for P, tag, name in ((8, "Half", "32\\,s"), (15, "One", "1\\,min"), (38, "Long", "2.5\\,min")):
        r = p10[f"P{P}"]
        fl, fp = r["full_vs_label_only"], r["full_vs_persist"]
        macros.update({f"fcProbe{tag}Acc": pct(fl["b_acc"]), f"fcProbe{tag}QWK": q(fl["b_qwk"]),
                       f"fcPersist{tag}Acc": pct(fp["a_acc"]), f"fcPersist{tag}QWK": q(fp["a_qwk"]),
                       f"fcProbe{tag}Gain": f"{100 * fp['diff_acc']:.1f}", f"fcProbe{tag}GainCI": pts(fp["ci_acc"])})
        assert fp["ci_acc"][0] > 0
        probe_lines.append(f"{name} & {pct(fp['a_acc'])} / {q(fp['a_qwk'])} & {pct(fl['a_acc'])} / {q(fl['a_qwk'])} & "
                           f"{pct(fl['b_acc'])} / {q(fl['b_qwk'])} & $+${100 * fp['diff_acc']:.1f} {pts(fp['ci_acc'])} & "
                           f"{d(fp['diff_qwk'])} {ci(fp['ci_qwk'])} \\\\\n")
    macros["fcSensorsOnlyAcc"] = pct(p10["P8"]["sensors_only"]["acc"])
    macros["fcSensorsOnlyQWK"] = q(p10["P8"]["sensors_only"]["qwk"])
    macros["fcMajorityAll"] = pct(p10["majority_acc"])
    (T / "fullcorpus_probe.tex").write_text("".join(probe_lines))
    p10b = load(R / "v12_3enc" / "phase10b.json")
    ENC = (("resnet18", "Res", "ResNet-18"), ("dinov2_vits14", "Dino", "DINOv2"), ("hsemotion", "Expr", "Expression encoder"))
    assert sorted(p10b["encoders"]) == sorted(e for e, _, _ in ENC) and p10b["matched"] >= 0.99
    macros.update({"fcEncBaseAcc": pct(p10b["arms"]["base"]["acc"]), "fcEncBaseQWK": q(p10b["arms"]["base"]["qwk"])})
    enc_rows = []
    for e, tag, name in ENC:
        t = p10b["tests"][f"enc_{e}_vs_base"]
        assert not t["helps"]
        macros.update({f"fcEnc{tag}Acc": pct(t["b_acc"]), f"fcEnc{tag}QWK": q(t["b_qwk"])})
        enc_rows.append((f"{name} vs.\\ sensors (unseen class)", "10B", "pre-decl.", t["diff_qwk"], t["ci_qwk"], t["p_holm"]))
    sp = p10b["self_report_probe"]
    macros.update({"fcSelfExtAcc": pct(sp["self_vs_external"]["acc"]), "fcSelfExtQWK": q(sp["self_vs_external"]["qwk"]),
                   "fcSelfPersistHalfAcc": pct(sp["P8"]["persist_self"]["acc"]), "fcSelfPersistHalfQWK": q(sp["P8"]["persist_self"]["qwk"]),
                   "fcSelfFullHalfAcc": pct(sp["P8"]["full_self"]["acc"]), "fcSelfFullHalfQWK": q(sp["P8"]["full_self"]["qwk"]),
                   "fcExtPersistHalfAcc": pct(sp["P8"]["persist_ext"]["acc"]), "fcExtPersistHalfQWK": q(sp["P8"]["persist_ext"]["qwk"])})
    (T / "fullcorpus_numbers.tex").write_text(
        "% written by analysis_v3/paper_fullcorpus.py -- do not edit\n"
        + "".join(f"\\newcommand{{\\{k}}}{{{v}}}\n" for k, v in macros.items()))

    # ---- pre-declared tests (and the two exploratory ones, marked)
    rows = [
        ("Best fusion vs.\\ best single stream", "3b", "pre-decl.", g["d_diff"], g["d_ci"], g["d_p"]),
        ("Stream vs.\\ classmate-label control", "3b", "pre-decl.",
         None, cm[best_cm]["diff"], cm[best_cm]["diff_p"]),
        ("Search, richer orders vs.\\ paper features", "6", "explor.",
         rich["acc"]["diff_qwk"], rich["acc"]["diff_qwk_ci"], None),
        ("Hahn vs.\\ Tchebichef (face crop)", "7", "pre-decl.",
         fam_q["diff_qwk"], fam_q["ci_qwk"], fam_q["p_holm"]),
        ("Face crop vs.\\ whole frame", "7", "pre-decl.", crop["diff_qwk"], crop["ci_qwk"], crop["p_qwk"]),
        ("Causal GRU vs.\\ window only", "8", "pre-decl.",
         p8["gru_vs_none"]["diff_qwk"], p8["gru_vs_none"]["ci_qwk"], p8["gru_vs_none"]["p_qwk"]),
        ("Class context vs.\\ own (unseen class)", "9", "pre-decl.",
         p9["own_class_vs_own"]["diff_qwk"], p9["own_class_vs_own"]["ci_qwk"], p9["own_class_vs_own"]["p_qwk"]),
    ] + enc_rows
    lines = ["% written by analysis_v3/paper_fullcorpus.py -- do not edit\n"]
    for name, phase, kind, diff, c, pv in rows:
        lines.append(f"{name} & {phase} & {kind} & {d(diff) if diff is not None else '--'} & {ci(c)} & "
                     f"{p(pv) if pv is not None else '--'} & no \\\\\n")
    (T / "fullcorpus_tests.tex").write_text("".join(lines))

    # ---- evaluation protocols: two paired steps, each with its own fixed model
    prot = [
        ("A", "Random windows", "same student, same minutes", rnd["acc"][0], rnd["qwk"][0]),
        ("A", "Held-out students", "classmates, same lessons", pg["acc"][0], pg["qwk"][0]),
        ("B", "Held-out students", "classmates, same lessons", a10["pg5_own"]["acc"][0], a10["pg5_own"]["qwk"][0]),
        ("B", "Held-out class", "nothing", a10["logo_own"]["acc"][0], a10["logo_own"]["qwk"][0]),
        ("B$'$", "Held-out students", "classmates' labels (via class features)",
         a10["pg5_own_class"]["acc"][0], a10["pg5_own_class"]["qwk"][0]),
        ("B$'$", "Held-out class", "nothing",
         a10["logo_own_class"]["acc"][0], a10["logo_own_class"]["qwk"][0]),
    ]
    lines = ["% written by analysis_v3/paper_fullcorpus.py -- do not edit\n"]
    for i, (model, split, shares, acc, qwk) in enumerate(prot):
        if i in (2, 4):
            lines.append("\\midrule\n")
        lines.append(f"{model} & {split} & {shares} & {pct(acc)} & {q(qwk)} \\\\\n")
    (T / "fullcorpus_protocol.tex").write_text("".join(lines))
    print(f"wrote {len(macros)} macros, {len(rows)} test rows, {len(prot)} protocol rows")


if __name__ == "__main__":
    main()
