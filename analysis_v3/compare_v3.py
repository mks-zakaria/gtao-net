"""Who/when report and Gate 1 verdict for any set of gTAO runs.

    python compare_v3.py --run "v2=<dir>" --run "v3 fold=<dir>" ... [--gate] [--boot 1000]

Each <dir> holds <config>/seed<k>/oof.json. Per run it prints:
* label variance between persons (how "trait-like" the labels are);
* per config: QWK mean ± s.d., between-person r, within-person r, balanced accuracy;
* person-bootstrap 95% CIs for every single stream, the best fusion and the shuffle
  control, plus the paired CI of (best fusion - best single);
* factorial contrasts, Wilcoxon on config means, Holm-corrected;
* with --gate: the Gate 1 verdict as amended in PLAN_accuracy.md (2026-11-17,
  fixed before any v3 output was seen).
"""
import argparse
import json

import numpy as np

from whowhen import contrasts, label_between_share, load_runs, person_bootstrap, summarise

STREAMS = ("face", "facemesh", "head_pose", "physio")


def best_of(summary, prefix_filter):
    items = [(c, s) for c, s in summary.items() if prefix_filter(c)]
    return max(items, key=lambda kv: kv[1]["qwk"]) if items else (None, None)


def report(name, runs, n_boot, gate):
    summary = {c: summarise(seeds) for c, seeds in runs.items()}
    first = next(iter(runs.values()))[0]
    persons = len(np.unique(first["person"]))
    share_person = label_between_share(first["y"], first["person"])
    share_session = label_between_share(first["y"], first["subject"])
    print(f"\n{'=' * 100}\n{name}: {len(first['y'])} windows, {persons} persons | label variance "
          f"between persons {share_person:.0%} (between sessions {share_session:.0%})")
    print(f"{'config':28s} {'QWK':>15s} {'between':>8s} {'within':>8s} {'bal.acc':>8s}")
    order = sorted(summary, key=lambda c: (not c.startswith("unimodal"), -summary[c]["qwk"]))
    for c in order:
        s = summary[c]
        print(f"{c:28s} {s['qwk']:+.3f} ± {s['qwk_sd']:.3f} {s['between']:+8.2f} "
              f"{s['within']:+8.2f} {s['bacc']:8.3f}")

    singles = [c for c in summary if c.startswith("unimodal")]
    fusions = [c for c in summary if c.startswith(("gate_", "core_"))]
    best_single, _ = best_of(summary, lambda c: c in singles)
    best_fusion, _ = best_of(summary, lambda c: c in fusions)
    ci = {}
    targets = singles + [c for c in (best_fusion, "shuffle_core_skip1_drop1") if c in summary]
    print(f"\nperson-bootstrap 95% CI ({n_boot} draws, persons resampled):")
    for c in targets:
        ci[c] = person_bootstrap(runs[c], n_boot=n_boot)
        print(f"  {c:28s} QWK [{ci[c]['qwk'][0]:+.3f}, {ci[c]['qwk'][1]:+.3f}]   "
              f"within r [{ci[c]['within'][0]:+.3f}, {ci[c]['within'][1]:+.3f}]")
    if best_single and best_fusion:
        diff = person_bootstrap(runs[best_fusion], runs[best_single], n_boot=n_boot)["diff"]
        above = sum(summary[c]["qwk"] > summary[best_single]["qwk"] for c in fusions)
        print(f"  best fusion {best_fusion} - best single {best_single}: "
              f"{summary[best_fusion]['qwk'] - summary[best_single]['qwk']:+.3f} "
              f"CI [{diff[0]:+.3f}, {diff[1]:+.3f}] | fusions above best single: {above}/{len(fusions)}")

    rows = contrasts(summary)
    if rows:
        print("\ncontrasts (config-level Wilcoxon, Holm across factors):")
        for r in rows:
            print(f"  {r['factor']:15s} wins {r['wins']}/{r['pairs']}  median {r['median']:+.3f}  "
                  f"p {r['p']:.4f}  p_holm {r['p_holm']:.4f}")

    result = {"summary": summary, "ci": ci, "contrasts": rows}
    if gate:
        def best_stream(stream):
            cands = [c for c in singles if c.startswith(f"unimodal_{stream}_")]
            return max(cands, key=lambda c: summary[c]["qwk"]) if cands else None
        recovered = []
        for stream in ("face", "head_pose", "physio"):
            c = best_stream(stream)
            if c and summary[c]["qwk"] >= 0.10:
                lo = (ci.get(c) or person_bootstrap(runs[c], n_boot=n_boot))["qwk"][0]
                if lo > 0:
                    recovered.append(stream)
        within_ok = [c for c in singles if summary[c]["within"] >= 0.10]
        print(f"\nGATE 1 ({name}): (a) recovered with CI>0: {recovered or 'none'} -> "
              f"{'PASS' if len(recovered) >= 2 else 'FAIL'} | (b) streams with within r >= 0.10: "
              f"{within_ok or 'none'} -> {'PASS' if within_ok else 'FAIL'}")
        result["gate"] = {"a_recovered": recovered, "b_within": within_ok}
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run", action="append", required=True, help="NAME=DIR")
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--gate", action="store_true")
    ap.add_argument("--json", default=None, help="write all results to this file")
    args = ap.parse_args()
    out = {}
    for item in args.run:
        name, path = item.split("=", 1)
        out[name] = report(name, load_runs(path), args.boot, args.gate)
    if args.json:
        with open(args.json, "w") as handle:
            json.dump(out, handle, indent=1, default=float)


if __name__ == "__main__":
    main()
