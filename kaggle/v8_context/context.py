"""Phase 8: does time context help? (rules pre-declared in context/PLAN_accuracy.md "Phase 8")

Arms, identical towers and hyper-parameters (fixed in advance, no search):
  none   window-only model
  ema    causal exponential smoothing of `none` probabilities within the session
  gru    causal GRU over the student's previous windows in the session (deployable live)
  bigru  bidirectional GRU (offline only, exploratory)
Same 5 person-grouped folds as the v4/v6 cv5, 3 seeds, epochs chosen on an inner person
split of the training fold. No label is ever an input.
Writes summary.json and oof.npz (out-of-fold predictions per arm and seed).

Local smoke test: V8_FAST=1 V8_INPUT=<folder with corpus_full.npz> python context.py
"""
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold

FAST = os.environ.get("V8_FAST", "") == "1"
INPUT = os.environ.get("V8_INPUT", "/kaggle/input")
OUT = Path("/kaggle/working") if os.path.isdir("/kaggle/working") else Path(os.environ.get("V8_OUT", "v8_out"))
OUT.mkdir(parents=True, exist_ok=True)
SEEDS = (0,) if FAST else (0, 1, 2)
FACE_K, CROP_K, SEQ_T = 8, 8, 5
HIDDEN, EMB, D, DROPOUT, LR, WD = 512, 64, 128, 0.3, 3e-4, 1e-5
MAX_EPOCHS, PATIENCE, SESS_PER_BATCH = (4 if FAST else 60), 8, 8
ALPHAS = (0.0, 0.5, 0.7, 0.8, 0.9, 0.95)
ARMS = ("none", "gru", "bigru")
torch.set_num_threads(os.cpu_count() or 4)
START = time.time()


def log(msg):
    print(f"[{(time.time() - START) / 60:6.1f} min] {msg}", flush=True)


def find(name):
    for folder, _, names in os.walk(INPUT):
        if name in names:
            return Path(folder) / name
    return None


def person_of(s):
    parts = str(s).split("/")
    return f"{parts[0]}/{parts[2]}" if len(parts) == 3 else str(s)


def qwk(y, p, k=3):
    o = np.zeros((k, k))
    np.add.at(o, (y, p), 1)
    w = (np.subtract.outer(np.arange(k), np.arange(k)) ** 2) / (k - 1) ** 2
    e = np.outer(o.sum(1), o.sum(0)) / max(o.sum(), 1)
    return float(1 - (w * o).sum() / (w * e).sum()) if (w * e).sum() > 0 else 0.0


def scores(y, p):
    recall = [float((p[y == c] == c).mean()) for c in range(3)]
    return {"acc": float((y == p).mean()), "qwk": qwk(y, p), "balanced_acc": float(np.mean(recall)),
            "recall": recall}


# ----------------------------------------------------------------------------- data
def lead(x, n_values, seq, face=None, seq_t=SEQ_T, k=None):
    x = x.reshape(len(x), n_values, seq)[:, :, :min(seq_t, seq)]
    if face:
        x = x.reshape(len(x), face, face, x.shape[-1])[:, :k, :k, :]
    return np.ascontiguousarray(x.reshape(len(x), -1), dtype=np.float32)


def load():
    path = find("corpus_rich.npz") or find("corpus_full.npz")
    d = np.load(path, allow_pickle=True)
    log(f"corpus {path}")
    n = len(d["y"])
    seq_mesh = d["feat_facemesh"].shape[1] // (478 * 3)
    seq_phys = d["feat_physio"].shape[1] // 8
    face_k = int(round((d["feat_face"].shape[1] // seq_mesh) ** 0.5))
    streams = {"face": lead(d["feat_face"], face_k * face_k, seq_mesh, face_k, k=FACE_K),
               "facemesh": lead(d["feat_facemesh"], 478 * 3, seq_mesh),
               "head_pose": lead(d["feat_head_pose"], 3, seq_mesh),
               "physio": lead(d["feat_physio"], 8, seq_phys)}
    keep = np.ones(n, bool)
    for s in ("face", "facemesh", "head_pose", "physio"):
        keep &= d[f"avail_{s}"].astype(bool)
    crop_path = find("corpus_crop.npz")
    if crop_path is not None:
        c = np.load(crop_path)
        assert np.array_equal(c["keys"].astype(str), d["keys"].astype(str))
        streams["face_crop"] = lead(c["feat_crop_tcheb"], 16 * 16, 8, 16, k=CROP_K)
    keys = d["keys"].astype(str)
    data = {"keys": keys[keep], "y": d["y"][keep].astype(np.int64),
            "session": np.array([k.split("#")[0] for k in keys[keep]]),
            "t": np.array([float(k.split("#")[1]) for k in keys[keep]]),
            "x": {s: v[keep] for s, v in streams.items()}}
    data["persons"] = np.array([person_of(s) for s in data["session"]])
    if FAST:
        sessions = np.unique(data["session"])
        chosen = np.random.default_rng(0).choice(sessions, 60, replace=False)
        m = np.isin(data["session"], chosen)
        data = {k: ({s: v[m] for s, v in data["x"].items()} if k == "x" else data[k][m]) for k in data}
    log(f"{len(data['y'])} windows, {len(np.unique(data['session']))} sessions, "
        f"{len(np.unique(data['persons']))} persons, streams { {s: v.shape[1] for s, v in data['x'].items()} }")
    return data


def session_index(data, rows):
    """Rows grouped by session, each sorted by time; plus the scaled time gap per row."""
    rows = np.asarray(rows)
    groups = {}
    for r in rows:
        groups.setdefault(data["session"][r], []).append(r)
    seqs = []
    for rs in groups.values():
        rs = np.array(rs)[np.argsort(data["t"][rs])]
        seqs.append(rs)
    return seqs


def gaps(data, seq):
    g = np.diff(data["t"][seq], prepend=data["t"][seq[0]])
    return np.log1p(np.clip(g / 4.0 - 1.0, 0, 50)).astype(np.float32)


# ----------------------------------------------------------------------------- model
class Model(nn.Module):
    def __init__(self, dims, arm):
        super().__init__()
        self.names = list(dims)
        self.towers = nn.ModuleDict({s: nn.Sequential(nn.Linear(d, HIDDEN), nn.GELU(), nn.Dropout(DROPOUT),
                                                      nn.Linear(HIDDEN, EMB), nn.LayerNorm(EMB))
                                     for s, d in dims.items()})
        self.proj = nn.Sequential(nn.Dropout(DROPOUT), nn.Linear(EMB * len(dims) + 1, D), nn.GELU())
        self.arm = arm
        if arm in ("gru", "bigru"):
            self.rnn = nn.GRU(D, D, batch_first=True, bidirectional=(arm == "bigru"))
            self.head = nn.Linear(D * (2 if arm == "bigru" else 1) + D, 3)
        else:
            self.head = nn.Linear(D, 3)

    def forward(self, xs, gap, lengths):
        """xs[s]: (B, T, d); gap (B, T, 1) -> logits (B, T, 3)."""
        z = torch.cat([self.towers[s](xs[s]) for s in self.names] + [gap], dim=-1)
        h = self.proj(z)
        if self.arm == "none":
            return self.head(h)
        packed = nn.utils.rnn.pack_padded_sequence(h, lengths, batch_first=True, enforce_sorted=False)
        out, _ = self.rnn(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True, total_length=h.shape[1])
        return self.head(torch.cat([out, h], dim=-1))


def batches(data, seqs, norm, order):
    for i in range(0, len(order), SESS_PER_BATCH):
        chunk = [seqs[j] for j in order[i:i + SESS_PER_BATCH]]
        T = max(len(s) for s in chunk)
        xs = {s: torch.zeros(len(chunk), T, data["x"][s].shape[1]) for s in data["x"]}
        gap = torch.zeros(len(chunk), T, 1)
        y = torch.full((len(chunk), T), -100, dtype=torch.long)
        for b, seq in enumerate(chunk):
            for s in data["x"]:
                xs[s][b, :len(seq)] = torch.from_numpy((data["x"][s][seq] - norm[s][0]) / norm[s][1])
            gap[b, :len(seq), 0] = torch.from_numpy(gaps(data, seq))
            y[b, :len(seq)] = torch.from_numpy(data["y"][seq])
        yield chunk, xs, gap, torch.tensor([len(s) for s in chunk]), y


def predict(model, data, seqs, norm):
    model.eval()
    probs = {}
    with torch.no_grad():
        for chunk, xs, gap, lengths, _ in batches(data, seqs, norm, np.arange(len(seqs))):
            p = F.softmax(model(xs, gap, lengths), dim=-1).numpy()
            for b, seq in enumerate(chunk):
                for j, r in enumerate(seq):
                    probs[r] = p[b, j]
    return probs


def train(data, arm, train_rows, val_rows, seed):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    norm = {s: (v[train_rows].mean(0), v[train_rows].std(0) + 1e-6) for s, v in data["x"].items()}
    tr_seqs, va_seqs = session_index(data, train_rows), session_index(data, val_rows)
    model = Model({s: v.shape[1] for s, v in data["x"].items()}, arm)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    counts = np.bincount(data["y"][train_rows], minlength=3) + 1.0
    w = torch.tensor((1 / np.sqrt(counts)) / (1 / np.sqrt(counts)).mean(), dtype=torch.float32)
    best, best_state, best_epoch = -1e9, None, 0
    yv = data["y"][np.concatenate(va_seqs)]
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        for _, xs, gap, lengths, y in batches(data, tr_seqs, norm, rng.permutation(len(tr_seqs))):
            opt.zero_grad()
            loss = F.cross_entropy(model(xs, gap, lengths).reshape(-1, 3), y.reshape(-1), weight=w,
                                   ignore_index=-100)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        pv = predict(model, data, va_seqs, norm)
        value = qwk(yv, np.array([pv[r].argmax() for r in np.concatenate(va_seqs)]))
        if value > best + 1e-9:
            best, best_epoch = value, epoch
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        if epoch - best_epoch >= PATIENCE:
            break
    model.load_state_dict(best_state)
    return model, norm, best_epoch, best


def ema(data, probs, rows, alpha):
    out = {}
    for seq in session_index(data, rows):
        s = None
        for r in seq:
            s = probs[r] if s is None else alpha * s + (1 - alpha) * probs[r]
            out[r] = s
    return out


def main():
    data = load()
    y, persons = data["y"], data["persons"]
    everything = np.arange(len(y))
    folds = list(StratifiedGroupKFold(5, shuffle=True, random_state=0).split(everything, y, persons))
    oof = {f"{a}_seed{s}": np.full(len(y), -1) for a in ARMS + ("ema",) for s in SEEDS}
    info = {"epochs": {}, "inner_val_qwk": {}, "alpha": {}}
    for fold, (tr, te) in enumerate(folds):
        inner = GroupShuffleSplit(1, test_size=0.2, random_state=fold)
        i_tr, i_va = next(inner.split(tr, y[tr], persons[tr]))
        i_tr, i_va = tr[i_tr], tr[i_va]
        te_seqs = session_index(data, te)
        for seed in SEEDS:
            for arm in ARMS:
                model, norm, ep, val = train(data, arm, i_tr, i_va, seed)
                pt = predict(model, data, te_seqs, norm)
                oof[f"{arm}_seed{seed}"][te] = [pt[r].argmax() for r in te]
                info["epochs"][f"{arm}_f{fold}_s{seed}"] = ep
                info["inner_val_qwk"][f"{arm}_f{fold}_s{seed}"] = round(val, 4)
                if arm == "none":     # ema: alpha chosen on the inner validation persons
                    pv = predict(model, data, session_index(data, i_va), norm)

                    def ema_qwk(a):
                        e = ema(data, pv, i_va, a)
                        return qwk(y[i_va], np.array([e[r].argmax() for r in i_va]))
                    best_alpha = max(ALPHAS, key=ema_qwk)
                    info["alpha"][f"f{fold}_s{seed}"] = best_alpha
                    pe = ema(data, pt, te, best_alpha)
                    oof[f"ema_seed{seed}"][te] = [pe[r].argmax() for r in te]
                s = scores(y[te], oof[f"{arm}_seed{seed}"][te])
                log(f"fold {fold} seed {seed} {arm}: epochs {ep}, inner val qwk {val:.3f}, "
                    f"test acc {s['acc']:.3f} qwk {s['qwk']:+.3f}")
    summary = {"n_windows": int(len(y)), "persons": int(len(np.unique(persons))),
               "streams": {s: int(v.shape[1]) for s, v in data["x"].items()},
               "majority_acc": float(np.mean(y == np.bincount(y).argmax())), "arms": {}, "info": info}
    for arm in ARMS + ("ema",):
        per_seed = [scores(y, oof[f"{arm}_seed{s}"]) for s in SEEDS]
        summary["arms"][arm] = {m: [float(np.mean([r[m] for r in per_seed])), float(np.std([r[m] for r in per_seed]))]
                                for m in ("acc", "qwk", "balanced_acc")}
        summary["arms"][arm]["recall"] = np.mean([r["recall"] for r in per_seed], axis=0).round(4).tolist()
        log(f"POOLED {arm}: {summary['arms'][arm]}")
    np.savez_compressed(OUT / "oof.npz", keys=data["keys"], y=y, persons=persons, session=data["session"],
                        **oof)
    summary["wall_h"] = round((time.time() - START) / 3600, 2)
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary["arms"], indent=1))


if __name__ == "__main__":
    main()
