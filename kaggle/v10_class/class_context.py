"""Phase 9: does class context help? (rules pre-declared in context/PLAN_accuracy.md "Phase 9")

Class features = mean of the OTHER students' features in the same class moment (group/
experiment + 4 s slot, the rule of the label-only class-moment prior), per stream, plus the
number of classmates. No label of a classmate is ever an input.
Confirmatory protocol: leave-one-class-out (3 groups); epochs by inner leave-one-group-out
on the 2 training groups, then retrain on both. Arms: own, own_class, class_only.
Exploratory: person-grouped 5-fold (1 seed) for own vs own_class (leak-prone protocol).
Data loading reuses kaggle/v8_context/context.py (embedded verbatim by make_kernel.py).
Writes summary.json and oof.npz.
"""
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold

CONTEXT_SRC = None   # make_kernel.py: source of kaggle/v8_context/context.py
HERE = Path(__file__).resolve().parent
if CONTEXT_SRC is not None:
    (Path("/tmp") / "v8_context.py").write_text(CONTEXT_SRC)
    ctx_path = Path("/tmp") / "v8_context.py"
else:
    ctx_path = HERE.parent / "v8_context" / "context.py"
spec = importlib.util.spec_from_file_location("v8_context", ctx_path)
C = importlib.util.module_from_spec(spec)
spec.loader.exec_module(C)

FAST = C.FAST
OUT = C.OUT
SEEDS = C.SEEDS
HIDDEN, EMB, D, DROPOUT, LR, WD, BATCH = 512, 64, 128, 0.3, 3e-4, 1e-5, 256
MAX_EPOCHS, PATIENCE = (3 if FAST else 60), 8
START = time.time()
log = C.log


def add_class_features(data):
    """x['class_<s>'] = mean over the other students in the same class moment; n_classmates."""
    session = data["session"]
    cls = np.array(["/".join(s.split("/")[:2]) for s in session])
    t0 = {c: data["t"][cls == c].min() for c in np.unique(cls)}
    slot = np.array([int(round((t - t0[c]) / 4.0)) for t, c in zip(data["t"], cls)])
    moment = np.char.add(np.char.add(cls, "#"), slot.astype(str))
    groups = {}
    for i, m in enumerate(moment):
        groups.setdefault(m, []).append(i)
    n = len(moment)
    count = np.zeros(n, np.float32)
    own = list(data["x"])
    for s in own:
        x = data["x"][s]
        out = np.zeros_like(x)
        for rows in groups.values():
            rows = np.array(rows)
            if len(rows) < 2:
                continue
            total = x[rows].sum(0)
            out[rows] = (total[None, :] - x[rows]) / (len(rows) - 1)   # leave this student out
        data["x"][f"class_{s}"] = out
    for rows in groups.values():
        count[rows] = len(rows) - 1
    data["x"]["class_count"] = count[:, None]
    data["group"] = np.array([s.split("/")[0] for s in session])
    log(f"class moments: {len(groups)}, classmates per window median {np.median(count):.0f}, "
        f"windows without classmates {(count == 0).mean():.2%}")
    return own


class Net(nn.Module):
    def __init__(self, dims):
        super().__init__()
        self.names = list(dims)
        self.towers = nn.ModuleDict({s: nn.Sequential(nn.Linear(d, HIDDEN if d > 1 else 8), nn.GELU(),
                                                      nn.Dropout(DROPOUT),
                                                      nn.Linear(HIDDEN if d > 1 else 8, EMB), nn.LayerNorm(EMB))
                                     for s, d in dims.items()})
        self.head = nn.Sequential(nn.Dropout(DROPOUT), nn.Linear(EMB * len(dims), D), nn.GELU(), nn.Linear(D, 3))

    def forward(self, xs):
        return self.head(torch.cat([self.towers[s](xs[s]) for s in self.names], dim=-1))


def fit(data, streams, tr, ev, seed, epochs=None):
    """Train on rows tr; early-stop on ev by QWK unless epochs is fixed. Returns (pred_ev, best_epoch)."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    norm = {s: (data["x"][s][tr].mean(0), data["x"][s][tr].std(0) + 1e-6) for s in streams}
    t = lambda rows: {s: torch.from_numpy((data["x"][s][rows] - norm[s][0]) / norm[s][1]) for s in streams}
    xt, xe = t(tr), t(ev)
    ytr = torch.from_numpy(data["y"][tr])
    net = Net({s: data["x"][s].shape[1] for s in streams})
    opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=WD)
    counts = np.bincount(data["y"][tr], minlength=3) + 1.0
    w = torch.tensor((1 / np.sqrt(counts)) / (1 / np.sqrt(counts)).mean(), dtype=torch.float32)
    best, best_state, best_epoch = -1e9, None, 0
    for epoch in range(1, (epochs or MAX_EPOCHS) + 1):
        net.train()
        order = torch.from_numpy(rng.permutation(len(tr)))
        for i in range(0, len(order), BATCH):
            b = order[i:i + BATCH]
            opt.zero_grad()
            F.cross_entropy(net({s: v[b] for s, v in xt.items()}), ytr[b], weight=w).backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
        if epochs is not None:
            continue
        net.eval()
        with torch.no_grad():
            value = C.qwk(data["y"][ev], net(xe).argmax(1).numpy())
        if value > best + 1e-9:
            best, best_epoch = value, epoch
            best_state = {k: v.clone() for k, v in net.state_dict().items()}
        if epoch - best_epoch >= PATIENCE:
            break
    if best_state is not None:
        net.load_state_dict(best_state)
    net.eval()
    with torch.no_grad():
        return net(xe).argmax(1).numpy(), best_epoch


def run_protocol(data, arms, folds, inner_of, seeds, tag):
    y = data["y"]
    oof = {f"{tag}_{a}_seed{s}": np.full(len(y), -1) for a in arms for s in seeds}
    epochs_used = {}
    for f, (tr, te) in enumerate(folds):
        for seed in seeds:
            for arm, streams in arms.items():
                best = [fit(data, streams, i_tr, i_va, seed)[1] for i_tr, i_va in inner_of(tr)]
                e = max(1, int(round(np.mean(best))))
                pred, _ = fit(data, streams, tr, te, seed, epochs=e)
                oof[f"{tag}_{arm}_seed{seed}"][te] = pred
                epochs_used[f"{tag}_{arm}_f{f}_s{seed}"] = [best, e]
                s = C.scores(y[te], pred)
                log(f"{tag} fold {f} seed {seed} {arm}: epochs {best}->{e}, acc {s['acc']:.3f} qwk {s['qwk']:+.3f}")
    return oof, epochs_used


def main():
    data = C.load()
    own = add_class_features(data)
    cls_streams = [f"class_{s}" for s in own] + ["class_count"]
    arms = {"own": own, "own_class": own + cls_streams, "class_only": cls_streams}
    y, persons, group = data["y"], data["persons"], data["group"]
    everything = np.arange(len(y))
    groups = np.unique(group)
    logo = [(everything[group != g], everything[group == g]) for g in groups]

    def inner_logo(tr):          # leave one of the training groups out
        return [(tr[group[tr] != g], tr[group[tr] == g]) for g in np.unique(group[tr])]

    def inner_person(tr):        # exploratory protocol: 20% of training persons
        a, b = next(GroupShuffleSplit(1, test_size=0.2, random_state=0).split(tr, y[tr], persons[tr]))
        return [(tr[a], tr[b])]

    oof, ep = run_protocol(data, arms, logo, inner_logo, SEEDS, "logo")
    pg5 = list(StratifiedGroupKFold(5, shuffle=True, random_state=0).split(everything, y, persons))
    oof2, ep2 = run_protocol(data, {k: arms[k] for k in ("own", "own_class")}, pg5, inner_person,
                             SEEDS[:1], "pg5")
    oof.update(oof2)
    ep.update(ep2)
    summary = {"n_windows": int(len(y)), "persons": int(len(np.unique(persons))),
               "groups": {g: int((group == g).sum()) for g in groups},
               "majority_acc": float(np.mean(y == np.bincount(y).argmax())), "arms": {}, "epochs": ep}
    for name in sorted({k.rsplit("_seed", 1)[0] for k in oof}):
        runs = [v for k, v in oof.items() if k.rsplit("_seed", 1)[0] == name]
        per = [C.scores(y, p) for p in runs]
        summary["arms"][name] = {m: [float(np.mean([r[m] for r in per])), float(np.std([r[m] for r in per]))]
                                 for m in ("acc", "qwk", "balanced_acc")}
        summary["arms"][name]["recall"] = np.mean([r["recall"] for r in per], axis=0).round(4).tolist()
        summary["arms"][name]["per_group_qwk"] = {g: float(np.mean([C.qwk(y[group == g], p[group == g]) for p in runs]))
                                                  for g in groups}
        log(f"POOLED {name}: acc {summary['arms'][name]['acc'][0]:.4f} qwk {summary['arms'][name]['qwk'][0]:+.4f}")
    np.savez_compressed(OUT / "oof.npz", keys=data["keys"], y=y, persons=persons, group=group, **oof)
    summary["wall_h"] = round((time.time() - START) / 3600, 2)
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
