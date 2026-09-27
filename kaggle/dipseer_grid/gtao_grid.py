# %% [markdown]
# # gTAO-Net grid v1 — full attention grid, subject-disjoint
#
# The real experiment. This run covers the attention grid of the draft
# (`DRAFT_TAONet-G_multimodal.md` §5) over the full 3-loss factorial:
#
#     {gate, CORE-attention} × {global skip ON, OFF} × {modality dropout ON, OFF}
#     × {CORAL, CORE-hybrid CE, soft-target} × GroupKFold-5 by subject × seeds {0,1,2}
#     = 360 model fits
#
# Plus, in the same notebook and protocol:
# * unimodal baselines (each of the 4 streams alone, CORAL only) — the fair
#   comparator for the face-unimodal anchor, evaluated on the SAME complete-window
#   population as the fusion grid (tagged `_cw`);
# * a permuted-label shuffle control (labels shuffled within the complete
#   population, representative skip-ON config) checking |QWK| ≤ SHUFFLE_TOL.
#
# Protocol (anchored to the thesis, ch.4/5 + the 22-ku grid):
# * DIPSER D1 slice, 4 s windows, `filter_nulls` rater convention, 5→3 collapse,
#   plurality vote at window midpoint — byte-identical to the published pipelines.
# * Window population = **complete windows** (all present streams): a missing
#   stream drops the window, never zero-fills. Availability is logged per stream.
# * GroupKFold by subject (fold of a seed-shuffled subject list), 3 seeds.
# * Losses (all in the same CORAL survival head):
#   - `coral`: ordinal threshold BCE (logits are cumulative thresholds).
#   - `hybrid`: CE(class probs) + 0.5·ordinal-BCE + 0.1·neighbour-smoothing.
#   - `soft`: threshold BCE against *cumulative soft* rater fractions.
# * Checks: H3 first-layer gradient probe (skip ON vs OFF, at init), gate-α
#   trajectories, permuted-label shuffle control, seed- and fold-wise reporting.
#
# Deliberate simplifications (documented, constant across configs): unweighted
# losses, no re-weighting, Adam + global-norm clip 1.0, 4 s windows (the H2
# window sweep comes later), no masked-eval robustness probe yet (H4 is later).

# %%
import json
import math
import os
import tempfile
import time
import zipfile
from itertools import product
from pathlib import Path

import mpmath
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

# ---------------------------------------------------------------- config ----
ON_KAGGLE = os.path.isdir("/kaggle/working")
OUT = Path("/kaggle/working") if ON_KAGGLE else Path("/tmp/gtao_grid_out")
INPUT_DIR = Path("/kaggle/input/dipseer-paper1-slice")
RUNS = OUT / "gtao_grid_v1"
RUNS.mkdir(parents=True, exist_ok=True)
METRICS = OUT / "grid_v1_metrics.json"

FAST = os.environ.get("GRID_FAST", "") == "1"      # local synthetic smoke
WINDOW_S = 4.0                                     # annotation protocol window
FACE_SIZE = 64
FACE_K = 16                                        # 2-D Tchebichef order -> 16x16
SEQ_K = 6                                          # 1-D Tchebichef order across time
N_CLASSES = 3
EMB = 64
HID = 128
LR = 3e-3
EPOCHS = 8 if FAST else 40
BATCH = 32
PATIENCE = 8
MIN_EPOCHS = 5
N_FOLDS = 5
SEEDS = (0, 1, 2)
MDROP_P = 0.2

STREAMS = ("face", "facemesh", "head_pose", "physio")
FUSIONS = ("gate", "core")
SKIPS = (True, False)
MDROPS = (True, False)
LOSSES = ("coral", "hybrid", "soft")

PHYSIO_CHANNELS = ("HeartRate.bpm",
                   "Accel.x", "Accel.y", "Accel.z",
                   "Gyro.x", "Gyro.y", "Gyro.z",
                   "Light.lux")

FACE_ANCHOR_CH4 = 0.517      # thesis ch.4 face anchor, seed-0, subject-disjoint
LORO_CEILING = 0.230         # thesis inter-annotator ceiling (4 s windows)
SHUFFLE_TOL = 0.10           # permuted-label control: |QWK| must stay below this

if ON_KAGGLE:
    torch.cuda.manual_seed_all(0)
device = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", device, "| FAST:", FAST, "| fits:", (len(FUSIONS) * len(SKIPS)
      * len(MDROPS) * len(LOSSES) + len(STREAMS) + 1) * N_FOLDS * len(SEEDS))

# %%
import sklearn  # noqa: E402
from sklearn.metrics import cohen_kappa_score  # noqa: E402


def log_step(step, **payload):
    doc = json.loads(METRICS.read_text()) if METRICS.exists() else {}
    doc["status"] = "running"
    doc.setdefault("steps", {})[step] = {"ts": time.strftime("%Y-%m-%d %H:%M:%S"), **payload}
    METRICS.write_text(json.dumps(doc, indent=2))


def qwk(y_true, y_pred):
    if len(y_true) < 2 or len(np.unique(y_true)) < 2:
        return 0.0
    try:
        v = float(cohen_kappa_score(y_true, y_pred, weights="quadratic"))
    except ValueError:
        v = 0.0
    return 0.0 if not math.isfinite(v) else v

# %% [markdown]
# ## 1) Tchebichef basis + transforms (mpmath recurrence, cached)

# %%
def _rho(n, N):
    r = mpmath.mpf(N)
    for k in range(1, n + 1):
        r *= N * N - k * k
    return r / (2 * n + 1)


def tchebichef_basis(N, K):
    with mpmath.workdps(50):
        t = [[mpmath.mpf(0)] * N for _ in range(K)]
        for x in range(N):
            t[0][x] = mpmath.mpf(1)
        if K >= 2:
            for x in range(N):
                t[1][x] = mpmath.mpf(2 * x - N + 1)
        for n in range(2, K):
            a = mpmath.mpf(2 * n - 1)
            c = mpmath.mpf((n - 1) * (N * N - (n - 1) * (n - 1)))
            for x in range(N):
                t[n][x] = (a * (2 * x - N + 1) * t[n - 1][x] - c * t[n - 2][x]) / n
        B = np.zeros((N, K), dtype=np.float32)
        for n in range(K):
            norm = mpmath.sqrt(_rho(n, N))
            for x in range(N):
                B[x, n] = float(t[n][x] / norm)
    return B


_BASIS = {}


def basis(N, K):
    key = (N, K)
    if key not in _BASIS:
        _BASIS[key] = tchebichef_basis(N, K)
    return _BASIS[key]


def transform_2d(img, K):
    B = basis(img.shape[-1], K)
    return (B.T @ img) @ B


def transform_1d(x, K):
    B = basis(x.shape[-1], K)
    return x @ B

# %% [markdown]
# ## 2) Bundle readers (labels, metadata streams, watch sensors)
#
# Same conventions as the smoke / published pipelines: `filter_nulls` rater
# change-points, `collapse_5_to_3`, plurality at window midpoint.

# %%
def parse_ts(s):
    s = str(s).strip().replace(".", ":").replace("-", ":").replace("_", ":")
    p = [x for x in s.split(":") if x != ""]
    h, m = int(p[0]), int(p[1])
    sec = int(p[2]) if len(p) > 2 else 0
    us = float(p[3].ljust(6, "0")) / 1e6 if len(p) > 3 and p[3] else 0.0
    return h * 3600 + m * 60 + sec + us


HEAD_POSE_KEYS = ("pitch", "yaw", "roll")


def read_facemesh(meta):
    person = meta.get("person") or {}
    fm = (person.get("face") or {}).get("facemesh")
    if not isinstance(fm, list) or not fm:
        return None
    try:
        pts = np.array([[p["x"], p["y"], p["z"]] for p in fm], dtype=np.float32)
    except (KeyError, TypeError):
        return None
    return pts if pts.ndim == 2 and pts.shape[1] == 3 else None


def read_pose(meta):
    pose = (meta.get("person") or {}).get("pose")
    if not isinstance(pose, list) or not pose:
        return None
    try:
        pts = np.array([[p["x"], p["y"], p["z"]] for p in pose], dtype=np.float32)
    except (KeyError, TypeError):
        return None
    return pts if pts.ndim == 2 and pts.shape[1] == 3 else None


def read_head_pose(meta):
    hp = meta.get("head_pose") if isinstance(meta.get("head_pose"), dict) else None
    if hp is None:
        try:
            hp = meta["person"]["face"]["headpose"]["pose"]
        except (KeyError, TypeError):
            return None
    if not isinstance(hp, dict):
        return None
    try:
        return np.array([float(hp[k]) for k in HEAD_POSE_KEYS], dtype=np.float32)
    except (KeyError, TypeError, ValueError):
        return None


def norm_watch_channel(sensor, fields):
    s = sensor.lower()
    if "hr" in s or "heart" in s:
        return {"value0": "HeartRate.bpm"}
    if "accel" in s:
        return {f"value{i}": f"Accel.{'xyzw'[i]}" for i in range(min(len(fields), 4))}
    if "gyr" in s:
        return {f"value{i}": f"Gyro.{'xyzw'[i]}" for i in range(min(len(fields), 4))}
    if "rot" in s:
        return {f"value{i}": f"Rot.{'wxyz'[i]}" for i in range(min(len(fields), 4))}
    if "light" in s:
        return {"value0": "Light.lux"}
    return {f: f"{sensor}.{f}" for f in fields}


def read_watch_sensors(watch_dir):
    out = {}
    if not watch_dir.is_dir():
        return out
    for f in sorted(watch_dir.iterdir()):
        if "__" in f.name or f.suffix != ".json":
            continue
        try:
            blob = json.loads(f.read_text())
        except json.JSONDecodeError:
            continue
        data = blob.get("data") if isinstance(blob, dict) else None
        if isinstance(data, dict):
            for sensor, records in data.items():
                if not isinstance(records, list):
                    continue
                for s in records:
                    if not isinstance(s, dict):
                        continue
                    tkey = next((k for k in s if "time" in k.lower()), None)
                    if tkey is None:
                        continue
                    raw_t = s[tkey]
                    t = raw_t if isinstance(raw_t, (int, float)) else parse_ts(raw_t)
                    fields = sorted(k for k in s if k != tkey and isinstance(s[k], (int, float)))
                    mapping = norm_watch_channel(str(sensor), fields)
                    for vf in fields:
                        key = mapping.get(vf, f"{sensor}.{vf}")
                        ch = out.setdefault(key, ([], []))
                        ch[0].append(t)
                        ch[1].append(float(s[vf]))
            continue

        # legacy raw-download schema: {"Datos": {<type>: [{"time","...", <field>}]}}
        datos = blob.get("Datos") if isinstance(blob, dict) else None
        if not isinstance(datos, dict):
            continue
        for sensor_type, samples in datos.items():
            if not isinstance(samples, list):
                continue
            for s in samples:
                if not isinstance(s, dict):
                    continue
                tkey = next((k for k in s if k.lower() == "time"), None)
                if tkey is None:
                    continue
                t = s[tkey]
                t = t if isinstance(t, (int, float)) else parse_ts(t)
                mapping = norm_watch_channel(str(sensor_type), list(s.keys()))
                for field_name, val in s.items():
                    if field_name == tkey or not isinstance(val, (int, float)):
                        continue
                    key = mapping.get(field_name, f"{sensor_type}.{field_name}")
                    ch = out.setdefault(key, ([], []))
                    ch[0].append(t)
                    ch[1].append(float(val))
    result = {}
    for key, (ts, vs) in out.items():
        order = np.argsort(np.asarray(ts, dtype=np.float64), kind="mergesort")
        result[key] = (np.asarray(ts, dtype=np.float64)[order],
                       np.asarray(vs, dtype=np.float32)[order])
    return result


def read_rater_labels(labels_dir):
    out = {}
    if not labels_dir.is_dir():
        return out
    for f in sorted(labels_dir.iterdir()):
        if "__" in f.name or f.suffix != ".json":
            continue
        try:
            raw = json.loads(f.read_text())
        except json.JSONDecodeError:
            continue
        if not isinstance(raw, list):
            continue
        cps = []
        for e in raw:
            if not isinstance(e, dict) or "datetime" not in e:
                continue
            v = e.get("attention")
            if v is None:
                continue
            cps.append((parse_ts(e["datetime"]), int(v)))
        out[f.stem] = sorted(cps, key=lambda r: r[0])
    return out


def value_at(cps, t):
    best = None
    for ts, v in cps:
        if ts <= t:
            best = v
        else:
            break
    return best


def collapse_5_to_3(v):
    return 0 if v <= 2 else (1 if v == 3 else 2)


def majority_vote(votes, n_classes=N_CLASSES):
    vs = [v for v in votes if v is not None]
    if not vs:
        return None
    return int(np.bincount(np.asarray(vs, dtype=int), minlength=n_classes).argmax())

# %% [markdown]
# ## 3) Corpus encoding — one subject at a time
#
# Each bundle (`.zipbin`, ~1 GB) is extracted to a temp dir, its subject decoded
# into per-window per-stream Tchebichef tensors (+ plurality label + soft rater
# target), then the bundle is deleted. Peak disk is one bundle.
#
# Soft target: per window, the 3-class distribution of collapsed rater votes at the
# midpoint (for the later soft-target loss factor; no loss is wasted now).

# %%
def pad_time(arr, k):
    if arr.shape[0] >= k:
        return arr
    if arr.shape[0] == 0:
        raise ValueError("empty time axis")
    return np.concatenate([arr, np.repeat(arr[-1:], k - arr.shape[0], axis=0)], axis=0)


def subject_windows(subject_root, group_key):
    """All 4 s windows with plurality label + soft rater target for one subject.

    ``group_key`` = "group_01/experiment_01/subject_01", from the bundle filename
    (the extracted tree has no group/experiment nesting).
    """
    frames = []
    for img in sorted((subject_root / "images").glob("*.png")):
        try:
            frames.append((parse_ts(img.stem), img))
        except (ValueError, IndexError):
            continue
    if not frames:
        return []
    raters = read_rater_labels(subject_root / "labels")
    if not raters:
        return []
    frame_ts = np.array([t for t, _ in frames], dtype=np.float64)
    t0, t1 = float(frame_ts[0]), float(frame_ts[-1])
    windows = []
    t = t0
    while t + WINDOW_S <= t1 + 1e-9:
        mid = t + WINDOW_S / 2
        votes = [collapse_5_to_3(value_at(cps, mid)) for cps in raters.values()
                 if value_at(cps, mid) is not None]
        if not votes:
            t += WINDOW_S
            continue
        label = majority_vote(votes)
        lo, hi = np.searchsorted(frame_ts, [t, t + WINDOW_S])
        idx = list(range(int(lo), int(hi)))
        if not idx:
            t += WINDOW_S
            continue
        group = group_key.split("/")[0]
        experiment = group_key.split("/")[1]
        subject = group_key.split("/")[2]
        soft = np.zeros(N_CLASSES, dtype=np.float32)
        for v in votes:
            soft[v] += 1.0
        soft /= soft.sum()
        windows.append({
            "key": f"{group}/{experiment}/{subject}#{t:.2f}",
            "subject": f"{group}/{experiment}/{subject}",
            "t_start": t, "t_end": t + WINDOW_S,
            "label": label, "soft": soft, "n_raters": len(votes),
            "frames": [frames[i] for i in idx]})
        t += WINDOW_S
    return windows


def encode_windows(windows, subject_root, physio):
    """Per-window per-stream feature tensor; a stream absent in a window -> None."""
    rows = []
    for w in windows:
        faces, fm_seq, hp_seq = [], [], []
        for _, img_path in w["frames"]:
            stem = img_path.stem
            img = np.asarray(Image.open(img_path).convert("L")
                             .resize((FACE_SIZE, FACE_SIZE)),
                             dtype=np.float32) / 255.0
            faces.append(transform_2d(img, FACE_K))
            meta_path = subject_root / "metadata" / f"{stem}.json"
            if not meta_path.exists():
                continue
            try:
                meta = json.loads(meta_path.read_text())
            except json.JSONDecodeError:
                continue
            fm = read_facemesh(meta)
            if fm is not None:
                fm_seq.append(fm)
            hp = read_head_pose(meta)
            if hp is not None:
                hp_seq.append(hp)

        if not faces:
            continue
        feat = {}
        face_seq = pad_time(np.stack(faces), SEQ_K)
        feat["face"] = transform_1d(
            np.moveaxis(face_seq, 0, -1).reshape(-1, face_seq.shape[0]), SEQ_K).ravel()
        if fm_seq:
            fm_t = pad_time(np.stack(fm_seq), SEQ_K)
            feat["facemesh"] = transform_1d(
                np.moveaxis(fm_t, 0, -1).reshape(478 * 3, fm_t.shape[0]), SEQ_K).ravel()
        if hp_seq:
            hp_t = pad_time(np.stack(hp_seq), SEQ_K)
            feat["head_pose"] = transform_1d(
                np.moveaxis(hp_t, 0, -1).reshape(3, hp_t.shape[0]), SEQ_K).ravel()

        ok = True
        ch_feats = []
        for ch in PHYSIO_CHANNELS:
            pts = physio.get(ch)
            if pts is None:
                ok = False
                break
            pts, vals = pts
            sel = (pts >= w["t_start"]) & (pts < w["t_end"])
            x = vals[sel]
            if len(x) < SEQ_K:
                if len(x) == 0:
                    ok = False
                    break
                x = np.pad(x, (0, SEQ_K - len(x)), mode="edge")
            ch_feats.append(transform_1d(x[None, :], SEQ_K).ravel())
        if ok:
            feat["physio"] = np.concatenate(ch_feats)

        row = {k: w[k] for k in ("key", "subject", "label", "soft", "n_raters")}
        row.update({k: None for k in STREAMS if k != "physio"})
        row["physio"] = None
        for k, v in feat.items():
            if k in STREAMS:
                row[k] = v
        rows.append(row)
    return rows


def iter_subject_dirs(root, workdir):
    """Yield ``(subject_root, bundle_stem)`` for every per-subject zipbin."""
    import shutil
    zips = sorted(root.glob("*.zipbin")) or sorted(root.glob("*.zip"))
    for z in zips:
        target = workdir / z.stem
        if target.exists():
            shutil.rmtree(target, ignore_errors=True)
        target.mkdir(parents=True, exist_ok=True)
        try:
            with zipfile.ZipFile(z) as zf:
                zf.extractall(target)
        except zipfile.BadZipFile:
            shutil.rmtree(target, ignore_errors=True)
            continue
        stem = z.stem
        for sub in sorted(p.parent for p in target.rglob("images") if p.is_dir()):
            yield sub, stem
        shutil.rmtree(target, ignore_errors=True)


def build_corpus(dataroot, workdir, cache, fast_max=None):
    """Encode every subject; cache the assembled arrays to ``cache``."""
    if cache.exists():
        print("loading cached corpus:", cache)
        return np.load(cache, allow_pickle=True)

    records = []
    seen = {}
    t0 = time.time()
    for sub, stem in iter_subject_dirs(dataroot, workdir):
        group_key = "/".join(stem.split("__")[:3])
        try:
            wins = subject_windows(sub, group_key)
            physio = read_watch_sensors(sub / "watch_sensors")
        except Exception as e:
            print(f"  ! {stem}: {type(e).__name__}: {e}", flush=True)
            continue
        if not wins:
            print(f"  - {stem}: no labelled windows", flush=True)
            continue
        rows = encode_windows(wins, sub, physio)
        if not rows:
            print(f"  - {stem}: face encode failed", flush=True)
            continue
        records.extend(rows)
        seen[stem] = len(rows)
        print(f"  + {stem}: {len(rows)} windows "
              f"[{group_key}] (elapsed {time.time() - t0:.0f}s)", flush=True)
        if fast_max and len(seen) >= fast_max:
            break

    if not records:
        raise SystemExit("no windows collected")

    keys = np.asarray([r["key"] for r in records])
    subjects = np.asarray([r["subject"] for r in records])
    y = np.asarray([r["label"] for r in records], dtype=np.int64)
    soft = np.stack([r["soft"] for r in records]).astype(np.float32)
    n_raters = np.asarray([r["n_raters"] for r in records], dtype=np.int64)
    avail = {s: np.asarray([r[s] is not None for r in records], dtype=bool)
             for s in STREAMS}
    feats = {s: None for s in STREAMS}
    for s in STREAMS:
        idx = [i for i, r in enumerate(records) if r[s] is not None]
        if idx:
            arr = np.asarray([records[i][s] for i in idx], dtype=np.float32)
            full = np.full((len(records), arr.shape[1]), np.nan, dtype=np.float32)
            full[idx] = arr
            feats[s] = full
        print(f"stream {s}: present in {len(idx)}/{len(records)} windows "
              f"({100 * len(idx) / (len(records) or 1):.1f}%)")
    empty2d = np.zeros((0, 0), dtype=np.float32)
    np.savez(cache, keys=keys, subjects=subjects, y=y, soft=soft,
             n_raters=n_raters, avail_face=avail["face"],
             avail_facemesh=avail["facemesh"], avail_head_pose=avail["head_pose"],
             avail_physio=avail["physio"],
             feat_face=feats["face"] if feats["face"] is not None else empty2d,
             feat_facemesh=feats["facemesh"] if feats["facemesh"] is not None else empty2d,
             feat_head_pose=feats["head_pose"] if feats["head_pose"] is not None else empty2d,
             feat_physio=feats["physio"] if feats["physio"] is not None else empty2d)
    print(f"corpus: {len(records)} windows, {len(seen)} subjects, "
          f"{time.time() - t0:.0f}s")
    return np.load(cache, allow_pickle=True)


def locate_dataset_root():
    for cand in ("/kaggle/input/dipseer-paper1-slice",
                 "/kaggle/input/datasets/dipseer-paper1-slice"):
        if any(Path(cand).glob("*.zipbin")):
            return Path(cand)
    for dp, _, fns in os.walk("/kaggle/input"):
        if any(f.endswith(".zipbin") for f in fns):
            return Path(dp)
    return None

# %% [markdown]
# ## 4) Grid model: gate vs CORE fusion, global skip, modality dropout, CORAL head
#
# * **gate**: cross-modal self-attention + a learned gate α that can drive a noisy
#   stream to ~0 (G-A: rejection).
# * **core**: the CORE-style fusion operationalised on DIPSER — self-attention over
#   modality embeddings followed by an *unweighted* pool. Re-weights attention but
#   cannot zero a stream (the claimed weakness).
# * **global skip**: per-stream linear maps from the raw Tchebichef vectors straight
#   to the pre-head vector; unattenuated head→input gradient path (G-C / H3).
# * **modality dropout**: train-time zeroing of whole tower outputs with probability
#   `p`, making fusion + gate learn from partial signals (H4 groundwork).

# %%
class StreamTower(nn.Module):
    def __init__(self, in_dim, hidden=HID, emb=EMB):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, hidden), nn.GELU(),
                                 nn.Linear(hidden, emb), nn.LayerNorm(emb))

    def forward(self, x):
        return self.net(x)


class GatedCrossModalFusion(nn.Module):
    def __init__(self, n_streams, emb=EMB):
        super().__init__()
        self.proj_q = nn.Linear(emb, emb, bias=False)
        self.proj_k = nn.Linear(emb, emb, bias=False)
        self.gate_net = nn.Sequential(nn.Linear(n_streams * emb, n_streams * emb),
                                      nn.GELU(), nn.Linear(n_streams * emb, n_streams))

    def forward(self, z):
        q = self.proj_q(z)
        k = self.proj_k(z)
        attn = torch.softmax(q @ k.transpose(1, 2) / math.sqrt(z.shape[-1]), dim=-1)
        z2 = attn @ z
        alpha = torch.softmax(self.gate_net(z2.reshape(z2.shape[0], -1)), dim=-1)
        fused = (alpha.unsqueeze(-1) * z2).sum(dim=1)
        return fused, alpha


class CoreAttentionFusion(nn.Module):
    """CORE-style: self-attention then an unweighted mean pool (no rejection)."""

    def __init__(self, n_streams, emb=EMB):
        super().__init__()
        self.proj_q = nn.Linear(emb, emb, bias=False)
        self.proj_k = nn.Linear(emb, emb, bias=False)

    def forward(self, z):
        q = self.proj_q(z)
        k = self.proj_k(z)
        attn = torch.softmax(q @ k.transpose(1, 2) / math.sqrt(z.shape[-1]), dim=-1)
        z2 = attn @ z
        return z2.mean(dim=1), None


class GlobalSkip(nn.Module):
    def __init__(self, stream_dims, emb=EMB):
        super().__init__()
        self.proj = nn.ModuleDict({k: nn.Linear(d, emb, bias=False)
                                   for k, d in stream_dims.items()})

    def forward(self, streams):
        return torch.stack([self.proj[k](x) for k, x in streams.items()]).sum(dim=0)


class CORALHead(nn.Module):
    def __init__(self, emb=EMB, num_classes=N_CLASSES):
        super().__init__()
        self.d = num_classes - 1
        self.w = nn.Linear(emb, 1)
        self.b = nn.Parameter(torch.linspace(0.5, -0.5, self.d))

    def forward(self, x):
        return self.w(x) + self.b

    def probs(self, logits):
        pgt = torch.sigmoid(logits)
        ext = torch.cat([torch.ones_like(pgt[:, :1]), pgt,
                         torch.zeros_like(pgt[:, :1])], dim=1)
        p = (ext[:, :-1] - ext[:, 1:]).clamp(min=0)
        return p / p.sum(dim=1, keepdim=True)

    def loss(self, logits, y):
        k = torch.arange(self.d, device=logits.device)[None, :]
        targets = (y[:, None] > k).float()
        return F.binary_cross_entropy_with_logits(logits, targets)

    def predict(self, logits):
        pgt = torch.sigmoid(logits)
        ext = torch.cat([torch.ones_like(pgt[:, :1]), pgt,
                         torch.zeros_like(pgt[:, :1])], dim=1)
        p = (ext[:, :-1] - ext[:, 1:]).clamp(min=0)
        return (p / p.sum(dim=1, keepdim=True)).argmax(dim=1)


def head_loss(head, logits, y, soft, loss="coral"):
    """Loss selector for the three factorial variants (same CORAL head)."""
    if loss == "coral":
        return head.loss(logits, y)
    if loss == "soft":
        cum = torch.cumsum(soft, dim=1)[:, :head.d]
        return F.binary_cross_entropy_with_logits(logits, 1.0 - cum)
    p = head.probs(logits)
    ce = -torch.log(p.gather(1, y[:, None]).squeeze(1) + 1e-8).mean()
    bce = head.loss(logits, y)
    q = torch.zeros_like(p)
    q.scatter_(1, y[:, None], 0.7)
    for o in (-1, 1):
        yn = y + o
        rows = torch.nonzero((yn >= 0) & (yn < N_CLASSES)).squeeze(1)
        if rows.numel():
            q[rows, yn[rows].clamp(0, N_CLASSES - 1)] += 0.15
    q = q / q.sum(dim=1, keepdim=True)
    smooth = -(q * torch.log(p + 1e-8)).sum(dim=1).mean()
    return ce + 0.5 * bce + 0.1 * smooth


class GTAONetGrid(nn.Module):
    def __init__(self, stream_dims, fusion="gate", use_skip=True, mdrop=False,
                 emb=EMB):
        super().__init__()
        self.streams = sorted(stream_dims)
        self.towers = nn.ModuleDict({k: StreamTower(stream_dims[k])
                                     for k in self.streams})
        self.fusion = (GatedCrossModalFusion(len(self.streams)) if fusion == "gate"
                       else CoreAttentionFusion(len(self.streams)))
        self.skip = GlobalSkip(stream_dims) if use_skip else None
        self.head = CORALHead()
        self.use_skip = use_skip
        self.mdrop = mdrop
        self.p = MDROP_P

    def forward(self, streams, apply_mdrop=None):
        z = torch.stack([self.towers[k](streams[k]) for k in self.streams], dim=1)
        if (self.training and self.mdrop) or apply_mdrop:
            mask = torch.rand(z.shape[0], z.shape[1], device=z.device) > self.p
            z = z * mask.unsqueeze(-1).float()
        fused, alpha = self.fusion(z)
        if self.use_skip and self.skip is not None:
            fused = fused + self.skip(streams)
        return self.head(fused), alpha

# %% [markdown]
# ## 5) Metrics + training helpers

# %%
def to_torch(streams_dict):
    return {k: torch.as_tensor(v, device=device) for k, v in streams_dict.items()}


@torch.no_grad()
def evaluate(model, Xv, yv):
    logits, _ = model(Xv, apply_mdrop=False)
    pred = model.head.predict(logits).cpu().numpy()
    y = yv.cpu().numpy()
    return {"qwk": qwk(y, pred),
            "acc": round(float((pred == y).mean()), 4),
            "mae": round(float(np.abs(pred - y).mean()), 4)}


def tower_first_grad_norm(model, streams, y, use_skip):
    model.use_skip = use_skip
    model.zero_grad()
    logits, _ = model(streams, apply_mdrop=False)
    model.head.loss(logits, y).backward()
    total = 0.0
    for tower in model.towers.values():
        w = tower.net[0].weight
        total += (w.grad ** 2).sum().item()
    return math.sqrt(total)


def skip_grad_norm(model, streams, y):
    model.use_skip = True
    model.zero_grad()
    logits, _ = model(streams, apply_mdrop=False)
    model.head.loss(logits, y).backward()
    total = 0.0
    for p in model.skip.proj.values():
        total += (p.weight.grad ** 2).sum().item()
    return math.sqrt(total)


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)


def fit_fold(model, Xtr, ytr, Xva, yva, stream_list, loss="coral", softtr=None):
    """Train one fold, early stop on val QWK, return (best_state, gate_trace, metas)."""
    set_seed(int(np.random.randint(2 ** 31)))           # fold-rooted seed
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    best = {"qwk": -1.0, "state": None, "epoch": 0}
    trace = []
    losses = []
    for epoch in range(1, EPOCHS + 1):
        model.train()
        perm = torch.randperm(len(ytr))
        for s in range(0, len(ytr), BATCH):
            idx = perm[s:s + BATCH]
            opt.zero_grad()
            logits, _ = model({k: Xtr[k][idx] for k in stream_list})
            y_l = ytr[idx]
            soft_l = softtr[idx] if softtr is not None else None
            loss_val = head_loss(model.head, logits, y_l, soft_l, loss)
            loss_val.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(loss_val.item())
        model.eval()
        va = evaluate(model, Xva, yva)
        if va["qwk"] > best["qwk"] - 1e-9:
            best = {"qwk": va["qwk"],
                    "state": {k: v.detach().clone() for k, v in model.state_dict().items()},
                    "epoch": epoch}
        if epoch % 5 == 0:
            with torch.no_grad():
                _, alpha = model({k: Xva[k] for k in stream_list}, apply_mdrop=False)
            gate = ([round(float(a), 4) for a in alpha.mean(dim=0).cpu().numpy()]
                    if alpha is not None else [])
            trace.append({"epoch": epoch, "val_qwk": va["qwk"], "gate": gate})
        if epoch >= MIN_EPOCHS and epoch - best["epoch"] >= PATIENCE:
            print(f"    early stop @ {epoch}, best val QWK {best['qwk']} "
                  f"(ep {best['epoch']})")
            break
    model.load_state_dict(best["state"])
    model.eval()
    return best, trace, {"epochs": epoch, "loss_last": round(float(np.mean(losses[-64:])), 4)}

# %% [markdown]
# ## 7) H4 probe: masked-eval (stream-drop robustness)
#
# gTAO-Net advertises that the gate **rejects** a broken/noisy stream (G-A) while the
# CORE fusion cannot. H4 tests this at *inference time*: re-train the two best fusion
# configs (`gate_skip1_drop0_soft`, `core_skip1_drop0_soft`), then at eval zero out
# each stream (simulates "sensor died") and measure the pooled-QWK degradation.
#
# If the rejection hypothesis holds, gate should degrade **less** than core when a
# stream is dropped, and its α should re-weight toward the surviving streams.

# %%
MASK_CFGS = (
    {"fusion": "gate", "use_skip": True, "mdrop": False, "loss": "soft"},
    {"fusion": "core", "use_skip": True, "mdrop": False, "loss": "soft"},
)


def run_masked_eval(corpus, seeds=(0,)):
    """Train the focal configs, evaluate with each stream zero-masked at eval time."""
    subset = np.arange(len(corpus["y"]))
    subjects = corpus["subjects"]
    ys = corpus["y"]
    soft_full = corpus["soft"]
    all_present = [s for s in STREAMS if corpus[f"feat_{s}"].shape[0] > 0]
    keep = np.ones(len(subset), dtype=bool)
    for s in all_present:
        keep &= corpus[f"avail_{s}"]
    idx_complete = subset[keep]
    print(f"H4 population: {len(idx_complete)}/{len(subset)} complete windows")

    per_seed = {}
    for seed in seeds:
        if FAST and seed != 0:
            continue
        uniq = sorted(set(subjects.tolist()))
        rng = np.random.default_rng(seed)
        rng.shuffle(uniq)
        fold_of_subj = {s: i % N_FOLDS for i, s in enumerate(uniq)}
        fold_of = np.array([fold_of_subj[s] for s in subjects])
        gate_rows = {}

        for cfg in MASK_CFGS:
            name = cfg_name(cfg)
            fdir = RUNS / name / f"seed{seed}"
            print(f"\n=== H4 masked-eval: {name} seed {seed} ===", flush=True)
            dims = {s: int(corpus[f"feat_{s}"].shape[1]) for s in all_present}
            cond_pred = {c: np.full(len(subset), np.nan)
                         for c in ["none"] + [f"mask_{s}" for s in all_present]}
            alpha_acc = {c: [] for c in cond_pred}
            for f in range(N_FOLDS):
                mtr, mte = fold_of[idx_complete] != f, fold_of[idx_complete] == f
                if len(set(ys[idx_complete][mtr])) < 2 or not mte.any():
                    continue
                idx_tr, idx_te = idx_complete[mtr], idx_complete[mte]
                Xtr = to_torch({s: corpus[f"feat_{s}"][idx_tr] for s in all_present})
                ytr = torch.as_tensor(ys[idx_tr], device=device)
                softtr = torch.as_tensor(soft_full[idx_tr], device=device)
                Xva = to_torch({s: corpus[f"feat_{s}"][idx_te] for s in all_present})
                yva = torch.as_tensor(ys[idx_te], device=device)

                set_seed(int(f"{seed}{f}"))
                model = GTAONetGrid(dims, fusion=cfg["fusion"],
                                    use_skip=cfg["use_skip"],
                                    mdrop=cfg["mdrop"]).to(device)
                best, _, metas = fit_fold(model, Xtr, ytr, Xva, yva, all_present,
                                          loss=cfg["loss"], softtr=softtr)
                model.load_state_dict(best["state"])
                model.eval()
                with torch.no_grad():
                    logits, alpha = model(Xva, apply_mdrop=False)
                    cond_pred["none"][idx_te] = (
                        model.head.predict(logits).cpu().numpy())
                    if alpha is not None:
                        alpha_acc["none"].append(alpha.cpu().numpy())
                    for s in all_present:
                        Xm = {k: (torch.zeros_like(v) if k == s else v)
                              for k, v in Xva.items()}
                        logits_m, alpha_m = model(Xm, apply_mdrop=False)
                        cond_pred[f"mask_{s}"][idx_te] = (
                            model.head.predict(logits_m).cpu().numpy())
                        if alpha_m is not None:
                            alpha_acc[f"mask_{s}"].append(alpha_m.cpu().numpy())
                print(f"  fold {f}: fit ep={metas['epochs']} "
                      f"val QWK={best['qwk']:.3f}", flush=True)

            out = {"config": cfg, "seed": seed, "conditions": {}}
            for c, pred in cond_pred.items():
                ok = np.isfinite(pred)
                yd = ys[ok].astype(int)
                pd = pred[ok].astype(int)
                entry = {"qwk": round(qwk(yd, pd), 4),
                         "acc": round(float(np.mean(pd == yd)), 4),
                         "mae": round(float(np.mean(np.abs(pd - yd))), 4),
                         "n": int(ok.sum())}
                if alpha_acc[c]:
                    a = np.concatenate(alpha_acc[c], axis=0)
                    entry["gate_alpha"] = [round(float(x), 4)
                                           for x in a.mean(axis=0)]
                out["conditions"][c] = entry
            gate_rows[name] = out
        per_seed[seed] = gate_rows

    print("\n=== H4 SUMMARY: pooled QWK by eval condition (mean±std over seeds) ===")
    avail_seeds = sorted(per_seed)
    names = [cfg_name(c) for c in MASK_CFGS]
    conds = ["none"] + [f"mask_{s}" for s in all_present]
    hdr = f"{'config':26s}"
    for c in conds:
        hdr += f" {c:>14s}"
    hdr += f" {'drop(max mask)':>15s}"
    print(hdr)
    summary = {}
    for name in names:
        line = f"{name:26s}"
        for c in conds:
            qs = [per_seed[s][name]["conditions"][c]["qwk"] for s in avail_seeds]
            line += f" {np.mean(qs):+.4f}±{np.std(qs):.3f}"
        base = np.mean([per_seed[s][name]["conditions"]["none"]["qwk"]
                        for s in avail_seeds])
        masks = ["mask_" + s for s in all_present]
        drops = [base - np.mean([per_seed[s][name]["conditions"][m]["qwk"]
                                 for s in avail_seeds]) for m in masks]
        line += f" {max(drops):>15.3f}"
        print(line)
        summary[name] = {c: {mtype: [per_seed[s][name]["conditions"][c].get(mtype)
                                     for s in avail_seeds]
                             for mtype in ("qwk", "acc", "mae")}
                         for c in conds}
        alpha0 = per_seed[0][name]["conditions"]["none"].get("gate_alpha")
        if alpha0:
            print(f"  gate α (none)   : {alpha0}")
            for m in masks:
                print(f"  gate α ({m:>10s}) : "
                      f"{per_seed[0][name]['conditions'][m].get('gate_alpha')}")
    return summary

# %%
def cfg_name(cfg):
    if cfg.get("unimodal"):
        return f"unimodal_{cfg['unimodal']}_cw"
    if cfg.get("shuffle"):
        return "shuffle_core_skip1_drop1"
    base = f"{cfg['fusion']}_skip{int(cfg['use_skip'])}_drop{int(cfg['mdrop'])}"
    return base if cfg.get("loss", "coral") == "coral" else base + f"_{cfg['loss']}"


def run_grid(corpus, seed, configs):
    subset = np.arange(len(corpus["y"]))
    subjects = corpus["subjects"]
    ys = corpus["y"]
    soft_full = corpus["soft"]

    uniq = sorted(set(subjects.tolist()))
    rng = np.random.default_rng(seed)
    rng.shuffle(uniq)
    fold_of_subj = {s: i % N_FOLDS for i, s in enumerate(uniq)}
    fold_of = np.array([fold_of_subj[s] for s in subjects])

    for cfg in configs:
        name = cfg_name(cfg)
        run_dir = RUNS / name / f"seed{seed}"
        run_dir.mkdir(parents=True, exist_ok=True)
        finish = run_dir / "oof.json"
        if finish.exists():
            print(f"cached {name} seed{seed} — skipping")
            continue
        if FAST and seed != 0:
            continue
        print(f"\n=== {name} seed {seed} ===", flush=True)
        out = {"config": cfg, "seed": seed, "folds": [], "gate_trace": {},
               "probe": None}

        if cfg.get("unimodal"):
            stream_list = [cfg["unimodal"]]
        else:
            stream_list = [s for s in STREAMS if corpus[f"feat_{s}"].shape[0] > 0]
        dims = {}
        for s in stream_list:
            arr = corpus[f"feat_{s}"]
            if arr.shape[0] > 0:
                dims[s] = arr.shape[1]
        if not dims or any(corpus[f"feat_{s}"].shape[0] == 0 for s in stream_list):
            print(f"  skipped {name} (empty stream features)")
            continue

        # Population: complete windows (all streams present) for EVERY config — the
        # fusion grid, unimodal baselines and shuffle control all share this mask.
        all_present = [s for s in STREAMS if corpus[f"feat_{s}"].shape[0] > 0]
        keep = np.ones(len(subset), dtype=bool)
        for s in all_present:
            keep &= corpus[f"avail_{s}"]
        idx_complete = subset[keep]
        print(f"complete windows: {len(idx_complete)}/{len(subset)}  "
              f"class dist: {np.bincount(ys[idx_complete], minlength=N_CLASSES)}")

        y_run = ys.copy()
        soft_run = soft_full.copy()
        if cfg.get("shuffle"):
            perm = rng.permutation(len(idx_complete))
            y_run[idx_complete] = ys[idx_complete][perm]
            soft_run[idx_complete] = soft_full[idx_complete][perm]
            print("shuffle control: labels permuted within complete population")

        oof_pred = np.full(len(subset), np.nan)
        tr_gate_trace = {}
        for f in range(N_FOLDS):
            mtr, mte = fold_of[idx_complete] != f, fold_of[idx_complete] == f
            if len(set(y_run[idx_complete][mtr])) < 2 or not mte.any():
                continue
            idx_tr = idx_complete[mtr]
            idx_te = idx_complete[mte]
            Xtr = to_torch({s: corpus[f"feat_{s}"][idx_tr] for s in stream_list})
            ytr = torch.as_tensor(y_run[idx_tr], device=device)
            softtr = (torch.as_tensor(soft_run[idx_tr], device=device)
                      if cfg.get("loss") == "soft" else None)
            Xva = to_torch({s: corpus[f"feat_{s}"][idx_te] for s in stream_list})
            yva = torch.as_tensor(y_run[idx_te], device=device)

            set_seed(int(f"{seed}{f}"))
            if cfg.get("unimodal"):
                model = GTAONetGrid(dims, fusion="core", use_skip=False,
                                    mdrop=False).to(device)
            else:
                model = GTAONetGrid(dims, fusion=cfg["fusion"],
                                    use_skip=cfg["use_skip"],
                                    mdrop=cfg["mdrop"]).to(device)

            if f == 0 and out["probe"] is None and cfg.get("use_skip"):
                k = min(64, len(idx_tr))
                px = {s: corpus[f"feat_{s}"][idx_tr[:k]] for s in stream_list}
                px = to_torch(px)
                py = torch.as_tensor(y_run[idx_tr[:k]], device=device)
                gon = tower_first_grad_norm(model, px, py, use_skip=True)
                goff = tower_first_grad_norm(model, px, py, use_skip=False)
                gskip = skip_grad_norm(model, px, py)
                model.use_skip = True
                out["probe"] = {"tower_first_skipON": round(gon, 4),
                                "tower_first_skipOFF": round(goff, 4),
                                "skip_grad": round(gskip, 4),
                                "ratio_skip_over_towerOFF":
                                    round(gskip / max(goff, 1e-9), 3)}

            t0 = time.time()
            best, trace, metas = fit_fold(model, Xtr, ytr, Xva, yva, stream_list,
                                          loss=cfg.get("loss", "coral"),
                                          softtr=softtr)
            model.load_state_dict(best["state"])
            with torch.no_grad():
                logits, _ = model(Xva, apply_mdrop=False)
                pred = model.head.predict(logits).cpu().numpy()
            oof_pred[idx_te] = pred
            tr_gate_trace[f] = trace
            out["folds"].append({
                "fold": f, "n_train": int(mtr.sum()), "n_val": int(mte.sum()),
                "best_val_qwk": best["qwk"], "best_epoch": best["epoch"],
                "val_class_dist": [int(c) for c in
                                   np.bincount(y_run[idx_te], minlength=N_CLASSES)],
                **metas, "wall_s": round(time.time() - t0, 1)})
            print(f"  fold {f}: val n={int(mte.sum())} best QWK={best['qwk']:.4f} "
                  f"ep={best['epoch']}  [{round(time.time() - t0, 1)}s]", flush=True)

        ok = np.isfinite(oof_pred)
        yd = y_run[ok].astype(int)
        pd = oof_pred[ok].astype(int)
        out["oof"] = {"keys": corpus["keys"][ok].astype(str).tolist(),
                      "y": yd.tolist(), "pred": pd.tolist(),
                      "subject": corpus["subjects"][ok].astype(str).tolist()}
        out["pooled"] = {"qwk": round(qwk(yd, pd), 4),
                         "acc": round(float(np.mean(pd == yd)), 4),
                         "mae": round(float(np.mean(np.abs(pd - yd))), 4),
                         "n": int(ok.sum()),
                         "n_subjects": len(set(corpus["subjects"][ok].tolist()))}
        out["gate_trace"] = {str(f): tr for f, tr in sorted(tr_gate_trace.items())}
        finish.write_text(json.dumps(out, indent=1))
        print(f"  POOLED QWK = {out['pooled']['qwk']} (n={ok.sum()}, "
              f"subjects={out['pooled']['n_subjects']})", flush=True)

# %%
def summarize(seeds, configs, write_to=METRICS):
    rows = []
    for cfg in configs:
        name = cfg_name(cfg)
        run_dir = RUNS / name
        qwks, accs, maes, ns, probes, gates = [], [], [], [], [], []
        for seed in seeds:
            f = run_dir / f"seed{seed}" / "oof.json"
            if not f.exists():
                continue
            d = json.loads(f.read_text())
            qwks.append(d["pooled"]["qwk"])
            accs.append(d["pooled"]["acc"])
            maes.append(d["pooled"]["mae"])
            ns.append(d["pooled"]["n"])
            probes.append(d.get("probe"))
            gates.append(d.get("gate_trace"))
        if not qwks:
            continue
        mean = float(np.mean(qwks)) if qwks else float("nan")
        std = float(np.std(qwks)) if len(qwks) > 1 else 0.0
        row = {"config": name, "fusion": cfg["fusion"] if not cfg.get("unimodal") else "unimodal",
               "skip": cfg.get("use_skip"), "mdrop": cfg.get("mdrop"),
               "loss": cfg.get("loss", "coral"), "unimodal": cfg.get("unimodal"),
               "shuffle": cfg.get("shuffle"),
               "qwk_mean": round(mean, 4), "qwk_std": round(std, 4),
               "qwk_per_seed": [round(q, 4) for q in qwks],
               "acc_mean": round(float(np.mean(accs)), 4),
               "mae_mean": round(float(np.mean(maes)), 4),
               "n_median": int(sorted(ns)[len(ns) // 2]),
               "probe_seed0": probes[0] if probes and probes[0] else None}
        # gate alphas at final epoch, seed 0, fold 0
        if gates and gates[0]:
            tr = sorted(gates[0].items())[0][1]
            if tr:
                last = tr[-1]["gate"]
                row["gate_alpha_final"] = last
        rows.append(row)

    rows.sort(key=lambda r: (-r["qwk_mean"], r["config"]))
    print("\n=== GRID v1 SUMMARY (full 3-loss factorial, subject-disjoint) ===")
    print(f"{'config':26s} {'QWK mean±std':>14s} {'acc':>7s} {'mae':>6s} "
          f"{'n':>5s}  probe(skip/tower)  gate_alpha")
    for r in rows:
        probe = r["probe_seed0"]
        pr = f"{probe['ratio_skip_over_towerOFF']:>10.1f}" if probe else "-"
        ga = r.get("gate_alpha_final")
        ga = "[" + ", ".join(f"{a:.2f}" for a in ga) + "]" if ga else "-"
        print(f"{r['config']:26s} {r['qwk_mean']:+.4f}±{r['qwk_std']:.3f} "
              f"{r['acc_mean']:7.3f} {r['mae_mean']:6.3f} {r['n_median']:5d} "
              f"{pr}  {ga}")
    print(f"\nanchors: face ch.4 = {FACE_ANCHOR_CH4}, LORO ceiling = {LORO_CEILING}, "
          f"majority = 0.0")

    shuffle_rows = [r for r in rows if r.get("shuffle")]
    if shuffle_rows:
        mx = max(abs(r["qwk_mean"]) for r in shuffle_rows)
        print(f"\nshuffle control: max |QWK| = {mx:.4f} "
              f"({'PASS' if mx <= SHUFFLE_TOL else 'FAIL'} ≤ {SHUFFLE_TOL})")
    unimodal_rows = [r for r in rows if r.get("unimodal")]
    if unimodal_rows:
        best = max(unimodal_rows, key=lambda r: r["qwk_mean"])
        print(f"best in-protocol unimodal: {best['config']} = {best['qwk_mean']:.4f}")

    doc = json.loads(write_to.read_text()) if write_to.exists() else {}
    doc["status"] = "summary"
    doc["summary"] = {
        "grid": "gtao_grid_v1 full 3-loss factorial",
        "n_fits": len(configs) * N_FOLDS * len(seeds),
        "rows": rows,
        "anchors": {"face_ch4": FACE_ANCHOR_CH4, "loro_ceiling": LORO_CEILING,
                    "majority_qwk": 0.0}}
    write_to.write_text(json.dumps(doc, indent=2))
    return rows

# %% [markdown]
# ## 7) Main

# %%
def main():
    t0 = time.time()
    tmp = Path(tempfile.mkdtemp())
    cache = OUT / "corpus_grid_v1.npz"

    dataroot = INPUT_DIR if ON_KAGGLE else Path(
        os.environ.get("DIPSEER_DATA_ROOT",
                       "/Users/mks/projects/research/phd/dipseer/kaggle"
                       "/paper2f/_synth_tmp/synth_slice"))
    if not ON_KAGGLE:
        for cand in (INPUT_DIR, dataroot):
            if cand.is_dir() and any(cand.glob("*.zipbin")):
                dataroot = cand
                break
    if dataroot.is_dir():
        print("data root:", dataroot)
    else:
        dataroot = locate_dataset_root()
        while dataroot is None and time.time() - t0 < 600:
            print("waiting for dataset mount ...")
            time.sleep(5)
            dataroot = locate_dataset_root()
    if dataroot is None:
        raise FileNotFoundError("no *.zipbin found")

    corpus = build_corpus(dataroot, tmp / "work", cache,
                          fast_max=(6 if FAST else None))

    ys = corpus["y"]
    keep = np.ones(len(ys), dtype=bool)
    for s in STREAMS:
        keep &= corpus[f"avail_{s}"]
    log_step("corpus", n_windows=len(ys), n_subjects=len(set(corpus["subjects"].tolist())),
             class_dist=[int(c) for c in np.bincount(ys, minlength=N_CLASSES)],
             complete_windows=int(keep.sum()),
             avail_pct={s: round(float(corpus[f"avail_{s}"].mean()), 3)
                        for s in STREAMS})

    configs = []
    for f, sk, md, lo in product(FUSIONS, SKIPS, MDROPS, LOSSES):
        configs.append({"fusion": f, "use_skip": sk, "mdrop": md, "loss": lo})
    for s in STREAMS:
        configs.append({"unimodal": s, "loss": "coral"})
    configs.append({"shuffle": True, "fusion": "core", "use_skip": True,
                    "mdrop": True, "loss": "coral"})
    for seed in SEEDS:
        run_grid(corpus, seed, configs)

    summarize(SEEDS, configs)
    h4 = run_masked_eval(corpus, seeds=SEEDS)
    h4_path = OUT / "h4_masked_eval.json"
    h4_path.write_text(json.dumps(h4, indent=2))

    doc = json.loads(METRICS.read_text())
    doc["status"] = "done"
    doc["finished_utc"] = time.strftime("%Y-%m-%d %H:%M:%S")
    doc["wall_s"] = round(time.time() - t0, 1)
    METRICS.write_text(json.dumps(doc, indent=2))
    print("total wall:", round(time.time() - t0, 1), "s")


if __name__ == "__main__":
    main()