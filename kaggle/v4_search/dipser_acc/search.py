"""v4: Bayesian hyper-parameter search (Optuna TPE) on Tchebichef features.

Goal (user, 2026-11-19): the best accuracy reachable with the Tchebichef front-end,
reported beside QWK, the metric that suits ordinal, imbalanced attention labels.

DATASET  dipser | daisee          OBJECTIVE  acc | qwk         (set by make_kernels.py)

Protocol (no test leakage):
* DAiSEE: official split. Trials train on Train and are scored on Validation; users that
  also appear in Validation/Test are removed from training. The best trial is retrained on
  Train + Validation (fixed epoch count = its best validation epoch, or the same GBM
  setting) and scores Test ONCE, for 3 seeds.
* DIPSER: no official split. A fixed person-disjoint, label-stratified split
  (StratifiedGroupKFold, seed 2026): test = fold 0 (20% of persons), validation = 20% of
  the remaining persons. Same search / retrain / test-once rule.
* Population: windows/clips where every stream exists (fixed across trials).
* Tchebichef orders are searched by SLICING the extracted moments (orthogonal basis:
  lower orders are the leading coefficients): face spatial order <= 16, temporal <= 6.

Search space: model {mlp, gbm}; streams (non-empty subset); orders; MLP: hidden, emb,
dropout, lr, weight decay, batch, head {coral, ce}, class weights {none, sqrt, balanced};
GBM (LightGBM): trees, leaves, lr, feature fraction, min child, class weights.
Writes study.json (every trial), final.json (test scores) and preds_test_seed<k>.json.
"""
import json
import os
import time
from pathlib import Path

import numpy as np

if "gbm" in os.environ.get("V4_MODELS", "mlp,gbm"):
    import lightgbm as lgb   # before torch; on macOS torch + lightgbm OpenMP clash (use V4_MODELS=mlp)
import torch
import torch.nn as nn
import torch.nn.functional as F

DATASET = os.environ.get("V4_DATASET", "dipser")
OBJECTIVE = os.environ.get("V4_OBJECTIVE", "acc")
TIMEOUT_S = float(os.environ.get("V4_TIMEOUT_S", str(3.2 * 3600)))
MAX_TRIALS = int(os.environ.get("V4_MAX_TRIALS", "400"))
FAST = os.environ.get("V4_FAST", "") == "1"
MODELS = os.environ.get("V4_MODELS", "mlp,gbm").split(",")   # "mlp" for macOS smoke tests
ON_KAGGLE = os.path.isdir("/kaggle/working")
OUT = Path("/kaggle/working") if ON_KAGGLE else Path(os.environ.get("V4_OUT", "v4_out"))
OUT.mkdir(parents=True, exist_ok=True)
FILES = {"dipser": "corpus_full.npz", "daisee": "daisee_clips_v1.npz"}
STREAMS = {"dipser": ("face", "facemesh", "head_pose", "physio"),
           "daisee": ("face", "facemesh", "head_pose")}[DATASET]
FACE_K, SEQ_K = 16, 6
FINAL_SEEDS = (0, 1, 2)
MAX_EPOCHS = 8 if FAST else 60
PATIENCE = 8
torch.set_num_threads(os.cpu_count() or 4)
LITERATURE = {"daisee": {"ViBED-Net": 73.43, "Abedi": 67.4, "Safa": 67.32,
                         "VisioPhysioENet": 63.09}, "dipser": {}}


# ----------------------------------------------------------------------------- metrics
def qwk(y, p, k):
    o = np.zeros((k, k))
    np.add.at(o, (y, p), 1)
    w = (np.subtract.outer(np.arange(k), np.arange(k)) ** 2) / (k - 1) ** 2
    e = np.outer(o.sum(1), o.sum(0)) / max(o.sum(), 1)
    denom = (w * e).sum()
    return float(1 - (w * o).sum() / denom) if denom > 0 else 0.0


def scores(y, p, k):
    recall = [float((p[y == c] == c).mean()) if (y == c).any() else None for c in range(k)]
    present = [r for r in recall if r is not None]
    return {"acc": float((y == p).mean()), "qwk": qwk(y, p, k),
            "balanced_acc": float(np.mean(present)), "recall": recall,
            "pred_dist": np.bincount(p, minlength=k).tolist()}


# ----------------------------------------------------------------------------- data
def find(name):
    explicit = os.environ.get("V4_CORPUS")
    if explicit and Path(explicit).is_file():
        return Path(explicit)
    for folder, _, names in os.walk("/kaggle/input"):
        if name in names:
            return Path(folder) / name
    raise FileNotFoundError(name)


def person_of(s):
    parts = str(s).split("/")
    return f"{parts[0]}/{parts[2]}" if len(parts) == 3 else str(s)


def load():
    d = np.load(find(FILES[DATASET]), allow_pickle=True)
    c = {k: d[k] for k in d.files}
    keep = np.ones(len(c["y"]), dtype=bool)
    for s in STREAMS:
        keep &= c[f"avail_{s}"].astype(bool)
    c = {k: (v[keep] if hasattr(v, "__len__") and len(v) == len(keep) else v)
         for k, v in c.items()}
    c["y"] = c["y"].astype(np.int64)
    k = int(c["y"].max()) + 1
    persons = np.array([person_of(s) for s in c["subjects"].astype(str)])
    if DATASET == "daisee":
        split = c["splits"].astype(str)
        test, val = split == "Test", split == "Validation"
        train = (split == "Train") & ~np.isin(persons, persons[val | test])
        trainval_final = (train | val) & ~np.isin(persons, persons[test])
    else:
        from sklearn.model_selection import StratifiedGroupKFold
        idx = np.arange(len(c["y"]))
        outer = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=2026)
        _, test_idx = next(outer.split(idx, c["y"], persons))
        test = np.isin(idx, test_idx)
        rest = idx[~test]
        inner = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=2026)
        _, val_part = next(inner.split(rest, c["y"][rest], persons[rest]))
        val = np.isin(idx, rest[val_part])
        train = ~test & ~val
        trainval_final = ~test
    if FAST:
        rng = np.random.default_rng(0)
        for mask in (train, val, test):
            on = np.flatnonzero(mask)
            mask[rng.permutation(on)[int(0.15 * len(on)):]] = False
        trainval_final = train | val
    c["persons"] = persons
    return c, k, {"train": train, "val": val, "test": test, "trainval": trainval_final}


def sliced(c, stream, face_k, seq_t):
    """Leading Tchebichef coefficients: (values, SEQ_K) layout, face values = 16x16."""
    x = c[f"feat_{stream}"]
    n_values = x.shape[1] // SEQ_K
    x = x.reshape(len(x), n_values, SEQ_K)[:, :, :seq_t]
    if stream == "face":
        x = x.reshape(len(x), FACE_K, FACE_K, seq_t)[:, :face_k, :face_k, :]
    return np.ascontiguousarray(x.reshape(len(x), -1), dtype=np.float32)


# ----------------------------------------------------------------------------- MLP
class Net(nn.Module):
    def __init__(self, dims, hidden, emb, dropout, k, head):
        super().__init__()
        self.towers = nn.ModuleList(
            nn.Sequential(nn.Linear(d, hidden), nn.GELU(), nn.Dropout(dropout),
                          nn.Linear(hidden, emb), nn.LayerNorm(emb)) for d in dims)
        self.drop = nn.Dropout(dropout)
        out = k - 1 if head == "coral" else k
        if head == "coral":
            self.fc = nn.Linear(emb * len(dims), 1, bias=False)
            self.bias = nn.Parameter(torch.linspace(1, -1, k - 1))
        else:
            self.fc = nn.Linear(emb * len(dims), out)
        self.head, self.k = head, k

    def forward(self, xs):
        z = self.drop(torch.cat([t(x) for t, x in zip(self.towers, xs)], dim=1))
        return self.fc(z) + self.bias if self.head == "coral" else self.fc(z)

    def predict(self, logits):
        if self.head == "coral":
            return (torch.sigmoid(logits) > 0.5).sum(1)
        return logits.argmax(1)


def class_weights(y, k, mode):
    counts = np.bincount(y, minlength=k).astype(float) + 1
    if mode == "none":
        w = np.ones(k)
    elif mode == "sqrt":
        w = 1 / np.sqrt(counts)
    else:
        w = 1 / counts
    return torch.tensor(w / w.mean() * 1.0, dtype=torch.float32)


def loss_fn(net, logits, y, w):
    if net.head == "coral":
        levels = (y[:, None] > torch.arange(net.k - 1)[None, :]).float()
        per = F.binary_cross_entropy_with_logits(logits, levels, reduction="none").sum(1)
        return (per * w[y]).mean()
    return F.cross_entropy(logits, y, weight=w)


def fit_mlp(p, xs_tr, y_tr, xs_ev, y_ev, k, seed, epochs=None):
    """Train; early-stop on (xs_ev, y_ev) by the OBJECTIVE unless epochs is fixed."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    mean = [x.mean(0, keepdims=True) for x in xs_tr]
    std = [x.std(0, keepdims=True) + 1e-6 for x in xs_tr]
    norm = lambda xs: [torch.as_tensor((x - m) / s) for x, m, s in zip(xs, mean, std)]
    tr, ev = norm(xs_tr), norm(xs_ev)
    ytr = torch.as_tensor(y_tr)
    net = Net([x.shape[1] for x in tr], p["hidden"], p["emb"], p["dropout"], k, p["head"])
    opt = torch.optim.AdamW(net.parameters(), lr=p["lr"], weight_decay=p["wd"])
    w = class_weights(y_tr, k, p["cw"])
    best, best_state, best_epoch = -1e9, None, 0
    for epoch in range(1, (epochs or MAX_EPOCHS) + 1):
        net.train()
        order = torch.randperm(len(ytr))
        for i in range(0, len(order), p["batch"]):
            b = order[i:i + p["batch"]]
            opt.zero_grad()
            loss_fn(net, net([x[b] for x in tr]), ytr[b], w).backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
        if epochs is not None:
            continue
        net.eval()
        with torch.no_grad():
            pred = net.predict(net(ev)).numpy()
        value = scores(y_ev, pred, k)[OBJECTIVE]
        if value > best + 1e-9:
            best, best_epoch = value, epoch
            best_state = {n: t.clone() for n, t in net.state_dict().items()}
        if epoch - best_epoch >= PATIENCE:
            break
    if best_state is not None:
        net.load_state_dict(best_state)
    net.eval()
    with torch.no_grad():
        pred = net.predict(net(ev)).numpy()
    return pred, best_epoch


# ----------------------------------------------------------------------------- GBM
def fit_gbm(p, x_tr, y_tr, x_ev, k, seed):
    counts = np.bincount(y_tr, minlength=k).astype(float) + 1
    w = {"none": np.ones(k), "sqrt": 1 / np.sqrt(counts), "balanced": 1 / counts}[p["cw"]]
    model = lgb.LGBMClassifier(
        n_estimators=p["trees"], num_leaves=p["leaves"], learning_rate=p["glr"],
        colsample_bytree=p["ff"], min_child_samples=p["min_child"], subsample=0.8,
        subsample_freq=1, reg_lambda=p["l2"], random_state=seed, n_jobs=os.cpu_count(),
        verbose=-1)
    model.fit(x_tr, y_tr, sample_weight=(w / w.mean())[y_tr])
    return model.predict(x_ev).astype(int)


# ----------------------------------------------------------------------------- search
def features(c, p, rows):
    xs = [sliced(c, s, p["face_k"], p["seq_t"])[rows] for s in p["streams"]]
    return xs


def suggest(trial):
    streams = [s for s in STREAMS if trial.suggest_categorical(f"use_{s}", [True, False])]
    if not streams:
        streams = [trial.suggest_categorical("fallback_stream", list(STREAMS))]
    p = {"streams": streams, "seq_t": trial.suggest_int("seq_t", 1, SEQ_K),
         "face_k": trial.suggest_categorical("face_k", [4, 6, 8, 12, 16]),
         "model": trial.suggest_categorical("model", MODELS),
         "cw": trial.suggest_categorical("cw", ["none", "sqrt", "balanced"])}
    if p["model"] == "mlp":
        p.update(hidden=trial.suggest_categorical("hidden", [64, 128, 256, 512]),
                 emb=trial.suggest_categorical("emb", [32, 64, 128]),
                 dropout=trial.suggest_float("dropout", 0.0, 0.5),
                 lr=trial.suggest_float("lr", 1e-4, 1e-2, log=True),
                 wd=trial.suggest_float("wd", 1e-6, 1e-1, log=True),
                 batch=trial.suggest_categorical("batch", [64, 128, 256, 512]),
                 head=trial.suggest_categorical("head", ["coral", "ce"]))
    else:
        p.update(trees=trial.suggest_int("trees", 100, 600, step=50),
                 leaves=trial.suggest_int("leaves", 8, 64, log=True),
                 glr=trial.suggest_float("glr", 0.01, 0.2, log=True),
                 ff=trial.suggest_float("ff", 0.05, 0.5, log=True),
                 min_child=trial.suggest_int("min_child", 10, 200, log=True),
                 l2=trial.suggest_float("l2", 1e-3, 10.0, log=True))
    return p


def main():
    import optuna
    start = time.time()
    c, k, split = load()
    y = c["y"]
    info = {"dataset": DATASET, "objective": OBJECTIVE, "classes": k,
            "n": {s: int(m.sum()) for s, m in split.items()},
            "persons": {s: int(len(np.unique(c["persons"][m]))) for s, m in split.items()},
            "class_dist": {s: np.bincount(y[m], minlength=k).tolist() for s, m in split.items()}}
    majority = int(np.bincount(y[split["trainval"]], minlength=k).argmax())
    info["majority_test"] = scores(y[split["test"]], np.full(split["test"].sum(), majority), k)
    info["literature_acc"] = LITERATURE[DATASET]
    print(json.dumps(info, indent=1), flush=True)
    tr, va = np.flatnonzero(split["train"]), np.flatnonzero(split["val"])
    epochs_of = {}

    def objective(trial):
        p = suggest(trial)
        t0 = time.time()
        xs_tr, xs_va = features(c, p, tr), features(c, p, va)
        if p["model"] == "mlp":
            pred, ep = fit_mlp(p, xs_tr, y[tr], xs_va, y[va], k, seed=trial.number)
            epochs_of[trial.number] = ep
        else:
            pred = fit_gbm(p, np.hstack(xs_tr), y[tr], np.hstack(xs_va), k, trial.number)
        s = scores(y[va], pred, k)
        for name, value in s.items():
            trial.set_user_attr(name, value)
        trial.set_user_attr("seconds", round(time.time() - t0, 1))
        print(f"trial {trial.number}: {p['model']} {p['streams']} -> val acc {s['acc']:.4f} "
              f"qwk {s['qwk']:+.4f} ({time.time() - t0:.0f}s)", flush=True)
        return s[OBJECTIVE]

    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=2026, n_startup_trials=20))
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study.optimize(objective, n_trials=6 if FAST else MAX_TRIALS,
                   timeout=None if FAST else TIMEOUT_S, gc_after_trial=True)
    trials = [{"number": t.number, "value": t.value, "params": t.params, **t.user_attrs}
              for t in study.trials if t.value is not None]
    (OUT / "study.json").write_text(json.dumps({"info": info, "trials": trials}, indent=1))

    best = study.best_trial
    p = suggest(optuna.trial.FixedTrial(best.params))
    final_rows, te = np.flatnonzero(split["trainval"]), np.flatnonzero(split["test"])
    xs_f, xs_te = features(c, p, final_rows), features(c, p, te)
    per_seed = []
    for seed in FINAL_SEEDS:
        if p["model"] == "mlp":
            pred, _ = fit_mlp(p, xs_f, y[final_rows], xs_te, y[te], k, seed,
                              epochs=max(1, epochs_of.get(best.number, 10)))
        else:
            pred = fit_gbm(p, np.hstack(xs_f), y[final_rows], np.hstack(xs_te), k, seed)
        s = scores(y[te], pred, k)
        per_seed.append(s)
        (OUT / f"preds_test_seed{seed}.json").write_text(json.dumps(
            {"keys": c["keys"][te].astype(str).tolist(), "y": y[te].tolist(),
             "pred": pred.tolist(), "subject": c["subjects"][te].astype(str).tolist()}))
        print(f"TEST seed {seed}: acc {s['acc']:.4f} qwk {s['qwk']:+.4f} "
              f"bal.acc {s['balanced_acc']:.4f} recall {s['recall']}", flush=True)
    summary = {m: [float(np.mean([s[m] for s in per_seed])), float(np.std([s[m] for s in per_seed]))]
               for m in ("acc", "qwk", "balanced_acc")}
    final = {"info": info, "best_trial": best.number, "best_val": best.value,
             "best_params": best.params, "final_epochs": epochs_of.get(best.number),
             "n_trials": len(trials), "test_per_seed": per_seed, "test_mean_sd": summary,
             "wall_s": round(time.time() - start, 1)}
    (OUT / "final.json").write_text(json.dumps(final, indent=1))
    print(json.dumps({"test_mean_sd": summary, "majority_test": info["majority_test"]["acc"],
                      "literature": LITERATURE[DATASET]}, indent=1))


if __name__ == "__main__":
    main()
