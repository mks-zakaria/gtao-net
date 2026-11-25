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

CONTEXT_SRC = '"""Phase 8: does time context help? (rules pre-declared in context/PLAN_accuracy.md "Phase 8")\n\nArms, identical towers and hyper-parameters (fixed in advance, no search):\n  none   window-only model\n  ema    causal exponential smoothing of `none` probabilities within the session\n  gru    causal GRU over the student\'s previous windows in the session (deployable live)\n  bigru  bidirectional GRU (offline only, exploratory)\nSame 5 person-grouped folds as the v4/v6 cv5, 3 seeds, epochs chosen on an inner person\nsplit of the training fold. No label is ever an input.\nWrites summary.json and oof.npz (out-of-fold predictions per arm and seed).\n\nLocal smoke test: V8_FAST=1 V8_INPUT=<folder with corpus_full.npz> python context.py\n"""\nimport json\nimport os\nimport time\nfrom pathlib import Path\n\nimport numpy as np\nimport torch\nimport torch.nn as nn\nimport torch.nn.functional as F\nfrom sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold\n\nFAST = os.environ.get("V8_FAST", "") == "1"\nINPUT = os.environ.get("V8_INPUT", "/kaggle/input")\nOUT = Path("/kaggle/working") if os.path.isdir("/kaggle/working") else Path(os.environ.get("V8_OUT", "v8_out"))\nOUT.mkdir(parents=True, exist_ok=True)\nSEEDS = (0,) if FAST else (0, 1, 2)\nFACE_K, CROP_K, SEQ_T = 8, 8, 5\nHIDDEN, EMB, D, DROPOUT, LR, WD = 512, 64, 128, 0.3, 3e-4, 1e-5\nMAX_EPOCHS, PATIENCE, SESS_PER_BATCH = (4 if FAST else 60), 8, 8\nALPHAS = (0.0, 0.5, 0.7, 0.8, 0.9, 0.95)\nARMS = ("none", "gru", "bigru")\ntorch.set_num_threads(os.cpu_count() or 4)\nSTART = time.time()\n\n\ndef log(msg):\n    print(f"[{(time.time() - START) / 60:6.1f} min] {msg}", flush=True)\n\n\ndef find(name):\n    for folder, _, names in os.walk(INPUT):\n        if name in names:\n            return Path(folder) / name\n    return None\n\n\ndef person_of(s):\n    parts = str(s).split("/")\n    return f"{parts[0]}/{parts[2]}" if len(parts) == 3 else str(s)\n\n\ndef qwk(y, p, k=3):\n    o = np.zeros((k, k))\n    np.add.at(o, (y, p), 1)\n    w = (np.subtract.outer(np.arange(k), np.arange(k)) ** 2) / (k - 1) ** 2\n    e = np.outer(o.sum(1), o.sum(0)) / max(o.sum(), 1)\n    return float(1 - (w * o).sum() / (w * e).sum()) if (w * e).sum() > 0 else 0.0\n\n\ndef scores(y, p):\n    recall = [float((p[y == c] == c).mean()) for c in range(3)]\n    return {"acc": float((y == p).mean()), "qwk": qwk(y, p), "balanced_acc": float(np.mean(recall)),\n            "recall": recall}\n\n\n# ----------------------------------------------------------------------------- data\ndef lead(x, n_values, seq, face=None, seq_t=SEQ_T, k=None):\n    x = x.reshape(len(x), n_values, seq)[:, :, :min(seq_t, seq)]\n    if face:\n        x = x.reshape(len(x), face, face, x.shape[-1])[:, :k, :k, :]\n    return np.ascontiguousarray(x.reshape(len(x), -1), dtype=np.float32)\n\n\ndef load():\n    path = find("corpus_rich.npz") or find("corpus_full.npz")\n    d = np.load(path, allow_pickle=True)\n    log(f"corpus {path}")\n    n = len(d["y"])\n    seq_mesh = d["feat_facemesh"].shape[1] // (478 * 3)\n    seq_phys = d["feat_physio"].shape[1] // 8\n    face_k = int(round((d["feat_face"].shape[1] // seq_mesh) ** 0.5))\n    streams = {"face": lead(d["feat_face"], face_k * face_k, seq_mesh, face_k, k=FACE_K),\n               "facemesh": lead(d["feat_facemesh"], 478 * 3, seq_mesh),\n               "head_pose": lead(d["feat_head_pose"], 3, seq_mesh),\n               "physio": lead(d["feat_physio"], 8, seq_phys)}\n    keep = np.ones(n, bool)\n    for s in ("face", "facemesh", "head_pose", "physio"):\n        keep &= d[f"avail_{s}"].astype(bool)\n    crop_path = find("corpus_crop.npz")\n    if crop_path is not None:\n        c = np.load(crop_path)\n        assert np.array_equal(c["keys"].astype(str), d["keys"].astype(str))\n        streams["face_crop"] = lead(c["feat_crop_tcheb"], 16 * 16, 8, 16, k=CROP_K)\n    keys = d["keys"].astype(str)\n    data = {"keys": keys[keep], "y": d["y"][keep].astype(np.int64),\n            "session": np.array([k.split("#")[0] for k in keys[keep]]),\n            "t": np.array([float(k.split("#")[1]) for k in keys[keep]]),\n            "x": {s: v[keep] for s, v in streams.items()}}\n    data["persons"] = np.array([person_of(s) for s in data["session"]])\n    if FAST:\n        sessions = np.unique(data["session"])\n        chosen = np.random.default_rng(0).choice(sessions, 60, replace=False)\n        m = np.isin(data["session"], chosen)\n        data = {k: ({s: v[m] for s, v in data["x"].items()} if k == "x" else data[k][m]) for k in data}\n    log(f"{len(data[\'y\'])} windows, {len(np.unique(data[\'session\']))} sessions, "\n        f"{len(np.unique(data[\'persons\']))} persons, streams { {s: v.shape[1] for s, v in data[\'x\'].items()} }")\n    return data\n\n\ndef session_index(data, rows):\n    """Rows grouped by session, each sorted by time; plus the scaled time gap per row."""\n    rows = np.asarray(rows)\n    groups = {}\n    for r in rows:\n        groups.setdefault(data["session"][r], []).append(r)\n    seqs = []\n    for rs in groups.values():\n        rs = np.array(rs)[np.argsort(data["t"][rs])]\n        seqs.append(rs)\n    return seqs\n\n\ndef gaps(data, seq):\n    g = np.diff(data["t"][seq], prepend=data["t"][seq[0]])\n    return np.log1p(np.clip(g / 4.0 - 1.0, 0, 50)).astype(np.float32)\n\n\n# ----------------------------------------------------------------------------- model\nclass Model(nn.Module):\n    def __init__(self, dims, arm):\n        super().__init__()\n        self.names = list(dims)\n        self.towers = nn.ModuleDict({s: nn.Sequential(nn.Linear(d, HIDDEN), nn.GELU(), nn.Dropout(DROPOUT),\n                                                      nn.Linear(HIDDEN, EMB), nn.LayerNorm(EMB))\n                                     for s, d in dims.items()})\n        self.proj = nn.Sequential(nn.Dropout(DROPOUT), nn.Linear(EMB * len(dims) + 1, D), nn.GELU())\n        self.arm = arm\n        if arm in ("gru", "bigru"):\n            self.rnn = nn.GRU(D, D, batch_first=True, bidirectional=(arm == "bigru"))\n            self.head = nn.Linear(D * (2 if arm == "bigru" else 1) + D, 3)\n        else:\n            self.head = nn.Linear(D, 3)\n\n    def forward(self, xs, gap, lengths):\n        """xs[s]: (B, T, d); gap (B, T, 1) -> logits (B, T, 3)."""\n        z = torch.cat([self.towers[s](xs[s]) for s in self.names] + [gap], dim=-1)\n        h = self.proj(z)\n        if self.arm == "none":\n            return self.head(h)\n        packed = nn.utils.rnn.pack_padded_sequence(h, lengths, batch_first=True, enforce_sorted=False)\n        out, _ = self.rnn(packed)\n        out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True, total_length=h.shape[1])\n        return self.head(torch.cat([out, h], dim=-1))\n\n\ndef batches(data, seqs, norm, order):\n    for i in range(0, len(order), SESS_PER_BATCH):\n        chunk = [seqs[j] for j in order[i:i + SESS_PER_BATCH]]\n        T = max(len(s) for s in chunk)\n        xs = {s: torch.zeros(len(chunk), T, data["x"][s].shape[1]) for s in data["x"]}\n        gap = torch.zeros(len(chunk), T, 1)\n        y = torch.full((len(chunk), T), -100, dtype=torch.long)\n        for b, seq in enumerate(chunk):\n            for s in data["x"]:\n                xs[s][b, :len(seq)] = torch.from_numpy((data["x"][s][seq] - norm[s][0]) / norm[s][1])\n            gap[b, :len(seq), 0] = torch.from_numpy(gaps(data, seq))\n            y[b, :len(seq)] = torch.from_numpy(data["y"][seq])\n        yield chunk, xs, gap, torch.tensor([len(s) for s in chunk]), y\n\n\ndef predict(model, data, seqs, norm):\n    model.eval()\n    probs = {}\n    with torch.no_grad():\n        for chunk, xs, gap, lengths, _ in batches(data, seqs, norm, np.arange(len(seqs))):\n            p = F.softmax(model(xs, gap, lengths), dim=-1).numpy()\n            for b, seq in enumerate(chunk):\n                for j, r in enumerate(seq):\n                    probs[r] = p[b, j]\n    return probs\n\n\ndef train(data, arm, train_rows, val_rows, seed):\n    torch.manual_seed(seed)\n    rng = np.random.default_rng(seed)\n    norm = {s: (v[train_rows].mean(0), v[train_rows].std(0) + 1e-6) for s, v in data["x"].items()}\n    tr_seqs, va_seqs = session_index(data, train_rows), session_index(data, val_rows)\n    model = Model({s: v.shape[1] for s, v in data["x"].items()}, arm)\n    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)\n    counts = np.bincount(data["y"][train_rows], minlength=3) + 1.0\n    w = torch.tensor((1 / np.sqrt(counts)) / (1 / np.sqrt(counts)).mean(), dtype=torch.float32)\n    best, best_state, best_epoch = -1e9, None, 0\n    yv = data["y"][np.concatenate(va_seqs)]\n    for epoch in range(1, MAX_EPOCHS + 1):\n        model.train()\n        for _, xs, gap, lengths, y in batches(data, tr_seqs, norm, rng.permutation(len(tr_seqs))):\n            opt.zero_grad()\n            loss = F.cross_entropy(model(xs, gap, lengths).reshape(-1, 3), y.reshape(-1), weight=w,\n                                   ignore_index=-100)\n            loss.backward()\n            nn.utils.clip_grad_norm_(model.parameters(), 1.0)\n            opt.step()\n        pv = predict(model, data, va_seqs, norm)\n        value = qwk(yv, np.array([pv[r].argmax() for r in np.concatenate(va_seqs)]))\n        if value > best + 1e-9:\n            best, best_epoch = value, epoch\n            best_state = {k: v.clone() for k, v in model.state_dict().items()}\n        if epoch - best_epoch >= PATIENCE:\n            break\n    model.load_state_dict(best_state)\n    return model, norm, best_epoch, best\n\n\ndef ema(data, probs, rows, alpha):\n    out = {}\n    for seq in session_index(data, rows):\n        s = None\n        for r in seq:\n            s = probs[r] if s is None else alpha * s + (1 - alpha) * probs[r]\n            out[r] = s\n    return out\n\n\ndef main():\n    data = load()\n    y, persons = data["y"], data["persons"]\n    everything = np.arange(len(y))\n    folds = list(StratifiedGroupKFold(5, shuffle=True, random_state=0).split(everything, y, persons))\n    oof = {f"{a}_seed{s}": np.full(len(y), -1) for a in ARMS + ("ema",) for s in SEEDS}\n    info = {"epochs": {}, "inner_val_qwk": {}, "alpha": {}}\n    for fold, (tr, te) in enumerate(folds):\n        inner = GroupShuffleSplit(1, test_size=0.2, random_state=fold)\n        i_tr, i_va = next(inner.split(tr, y[tr], persons[tr]))\n        i_tr, i_va = tr[i_tr], tr[i_va]\n        te_seqs = session_index(data, te)\n        for seed in SEEDS:\n            for arm in ARMS:\n                model, norm, ep, val = train(data, arm, i_tr, i_va, seed)\n                pt = predict(model, data, te_seqs, norm)\n                oof[f"{arm}_seed{seed}"][te] = [pt[r].argmax() for r in te]\n                info["epochs"][f"{arm}_f{fold}_s{seed}"] = ep\n                info["inner_val_qwk"][f"{arm}_f{fold}_s{seed}"] = round(val, 4)\n                if arm == "none":     # ema: alpha chosen on the inner validation persons\n                    pv = predict(model, data, session_index(data, i_va), norm)\n\n                    def ema_qwk(a):\n                        e = ema(data, pv, i_va, a)\n                        return qwk(y[i_va], np.array([e[r].argmax() for r in i_va]))\n                    best_alpha = max(ALPHAS, key=ema_qwk)\n                    info["alpha"][f"f{fold}_s{seed}"] = best_alpha\n                    pe = ema(data, pt, te, best_alpha)\n                    oof[f"ema_seed{seed}"][te] = [pe[r].argmax() for r in te]\n                s = scores(y[te], oof[f"{arm}_seed{seed}"][te])\n                log(f"fold {fold} seed {seed} {arm}: epochs {ep}, inner val qwk {val:.3f}, "\n                    f"test acc {s[\'acc\']:.3f} qwk {s[\'qwk\']:+.3f}")\n    summary = {"n_windows": int(len(y)), "persons": int(len(np.unique(persons))),\n               "streams": {s: int(v.shape[1]) for s, v in data["x"].items()},\n               "majority_acc": float(np.mean(y == np.bincount(y).argmax())), "arms": {}, "info": info}\n    for arm in ARMS + ("ema",):\n        per_seed = [scores(y, oof[f"{arm}_seed{s}"]) for s in SEEDS]\n        summary["arms"][arm] = {m: [float(np.mean([r[m] for r in per_seed])), float(np.std([r[m] for r in per_seed]))]\n                                for m in ("acc", "qwk", "balanced_acc")}\n        summary["arms"][arm]["recall"] = np.mean([r["recall"] for r in per_seed], axis=0).round(4).tolist()\n        log(f"POOLED {arm}: {summary[\'arms\'][arm]}")\n    np.savez_compressed(OUT / "oof.npz", keys=data["keys"], y=y, persons=persons, session=data["session"],\n                        **oof)\n    summary["wall_h"] = round((time.time() - START) / 3600, 2)\n    (OUT / "summary.json").write_text(json.dumps(summary, indent=1))\n    print(json.dumps(summary["arms"], indent=1))\n\n\nif __name__ == "__main__":\n    main()\n'
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
