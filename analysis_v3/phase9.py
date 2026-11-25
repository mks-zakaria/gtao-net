"""Phase 9 confirmatory analysis (rules pre-declared in context/PLAN_accuracy.md "Phase 9").

Input results/v10/oof.npz (kernel gtao-v10-dipser-class-context). Paired person bootstrap
(57 persons, 2000 draws) of pooled out-of-fold QWK, seeds averaged, two-sided p.
Confirmatory: leave-one-class-out own_class vs own ("helps" if CI excludes 0 and p < 0.05).
Exploratory: class_only vs own (leave-one-class-out); own_class vs own under person-grouped
5-fold (1 seed, leak-prone protocol); own under person-grouped vs leave-one-class-out.
Writes results/v10/phase9.json.
"""
import json
from pathlib import Path

import numpy as np

from phase8 import paired

R = Path(__file__).resolve().parent.parent / "results" / "v10"


def main():
    z = np.load(R / "oof.npz", allow_pickle=True)
    y, persons = z["y"], z["persons"].astype(str)
    runs = lambda name: [z[k] for k in sorted(z.files) if k.rsplit("_seed", 1)[0] == name]
    for name in ("logo_own", "logo_own_class", "logo_class_only", "pg5_own", "pg5_own_class"):
        assert runs(name) and all((p >= 0).all() for p in runs(name)), name
    report = {"own_class_vs_own": paired(y, persons, runs("logo_own"), runs("logo_own_class"), 0),
              "exploratory_class_only_vs_own": paired(y, persons, runs("logo_own"), runs("logo_class_only"), 1),
              "exploratory_pg5_own_class_vs_own": paired(y, persons, runs("pg5_own"), runs("pg5_own_class"), 2),
              "exploratory_own_logo_vs_pg5": paired(y, persons, runs("pg5_own"), runs("logo_own")[:1], 3)}
    r = report["own_class_vs_own"]
    report["class_context_helps"] = bool(r["p_qwk"] < 0.05 and r["ci_qwk"][0] > 0)
    (R / "phase9.json").write_text(json.dumps(report, indent=1))
    for name, r in report.items():
        if isinstance(r, dict):
            print(f"{name:36s} QWK {r['a_qwk']:.3f} -> {r['b_qwk']:.3f} diff {r['diff_qwk']:+.3f} {r['ci_qwk']} "
                  f"p {r['p_qwk']:.3f} | acc {r['a_acc']:.3f} -> {r['b_acc']:.3f} {r['ci_acc']}")
    print("class context helps:", report["class_context_helps"])


if __name__ == "__main__":
    main()
