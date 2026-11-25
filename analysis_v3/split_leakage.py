"""Random stratified split vs person-grouped split, same model (leakage diagnostic).

    V4_MODELS=mlp V4_CORPUS=<corpus_full.npz> python split_leakage.py
Uses the best accuracy-tuned setting of results/v4/dipser_acc with its fixed epoch count,
so neither scheme early-stops on its test fold. 5 folds each, seed 0, all complete windows.
"""
import importlib.util
import json
import os
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

ROOT = Path(__file__).resolve().parent.parent
os.environ.setdefault("V4_DATASET", "dipser")
spec = importlib.util.spec_from_file_location("s", ROOT / "kaggle/v4_search/search.py")
S = importlib.util.module_from_spec(spec)
spec.loader.exec_module(S)

final = json.loads((ROOT / "results/v4/dipser_acc/final.json").read_text())
import optuna
p = S.suggest(optuna.trial.FixedTrial(final["best_params"]))
epochs = max(1, final["final_epochs"] or 10)
c, k, _ = S.load()
S.set_orders(c)   # search.py >= da9f8fb reads the stored orders (paper corpus: 16 x 6, as before)
y, persons = c["y"], c["persons"]
X = S.features(c, p, np.arange(len(y)))
schemes = {"random stratified (windows)": StratifiedKFold(5, shuffle=True, random_state=0).split(y, y),
           "person-grouped stratified": StratifiedGroupKFold(5, shuffle=True, random_state=0).split(y, y, persons)}
out = {"params": p, "epochs": epochs, "n": int(len(y)), "persons": int(len(np.unique(persons)))}
for name, folds in schemes.items():
    rows = []
    for tr, te in folds:
        pred, _ = S.fit_mlp(p, [x[tr] for x in X], y[tr], [x[te] for x in X], y[te], k, 0,
                            epochs=epochs)
        rows.append(S.scores(y[te], pred, k))
        shared = len(set(persons[tr]) & set(persons[te]))
        print(f"{name}: acc {rows[-1]['acc']:.4f} qwk {rows[-1]['qwk']:+.4f} "
              f"(test persons also in train: {shared}/{len(set(persons[te]))})", flush=True)
    out[name] = {m: [float(np.mean([r[m] for r in rows])), float(np.std([r[m] for r in rows]))]
                 for m in ("acc", "qwk", "balanced_acc")}
    out[name]["recall"] = np.mean([r["recall"] for r in rows], axis=0).round(3).tolist()
    out[name]["majority_acc"] = float(np.mean(y == np.bincount(y).argmax()))
print(json.dumps({k: v for k, v in out.items() if k not in ("params",)}, indent=1))
(ROOT / "results/v4/split_leakage.json").write_text(json.dumps(out, indent=1))
