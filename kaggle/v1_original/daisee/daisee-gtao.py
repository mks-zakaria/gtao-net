# %% [markdown]
# # gTAO-Net on DAiSEE — 3-stream competitive benchmark (v1)
#
# Transfers the DIPSER/EngageNet gTAO-Net grid verbatim (same towers, fusion,
# CORAL head, losses, hyper-parameters, **identical stream dimensions**) onto
# **DAiSEE**, the canonical 4-level engagement benchmark (112 users, 8,571 fully
# labelled 10 s webcam clips), competing with published SOTA:
# ViBED-Net 73.43% / Safa supervised-contrastive ordinal 67.32% / Abedi ordinal
# TCN 67.4% / VisioPhysioENet 63.09% (all on the *official* split / test set).
#
# Streams (no rPPG on external sets — DIPSER keeps its 4th physio stream):
#   * face      — 2-D Tchebichef moments of the **MediaPipe face crop**, dim 1536
#   * facemesh  — MediaPipe 478 refined landmarks per frame, dim **8604**
#   * head_pose — solvePnP pitch/yaw/roll per frame, dim **18**
# Temporal axis: 1-D Tchebichef over a 6-frame window (`SEQ_K=6`); each 10 s clip
# contributes 6 uniformly sampled frames (the exact gtao window semantics).
#
# Two protocols (same as gtao_grid / EngageNet, both subject-disjoint):
#   * **P1 — official split**: train on `Train`, eval on `Validation` (users are
#     disjoint by construction); headline accuracy comparable with the listed
#     published numbers.
#   * **P2 — DIPSER-style**: full factorial
#     `{gate,core} × {skip ON,OFF} × {drop ON,OFF} × {coral,hybrid,soft}`
#     × GroupKFold-5 **by user** × seeds {0,1,2}, pooled OOF QWK/acc/MAE, plus
#     unimodal baselines, shuffle control and masked-eval (H4) — byte-identical
#     protocol to `gtao_grid.py` / `engagenet_gt.py`.
#
# Labels: DAiSEE `Engagement` column, ordinal 0..3 (we never quote the
# non-official Hossain 77.97% split number).

# %%
import collections
import csv
import json
import math
import os
import re
import tempfile
import time
from itertools import product
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

try:
    import cv2
except ImportError:  # local smoke without opencv
    cv2 = None

# %%
ON_KAGGLE = os.path.isdir("/kaggle/working")
if ON_KAGGLE:
    os.system("pip -q install mpmath scikit-learn opencv-python-headless 'protobuf==4.25.3' 'mediapipe==0.10.14'")
    import pathlib as _pl
    import google.protobuf as _gp
    _pb_pkg = _pl.Path(_gp.__file__).parent
    _rv = _pb_pkg / "runtime_version.py"
    if not _rv.exists():
        _rv.write_text(
            "import google.protobuf as _g\n"
            "from enum import IntEnum\n"
            "class Domain(IntEnum):\n"
            "    APP = 0\n    PUBLIC = 1\n    PLUGIN = 2\n    LOCAL = 3\n"
            "def get_runtime_version_info():\n"
            "    _maj, _min, _pat = _g.__version__.split('.')\n"
            "    return type('RuntimeVersionInfo', (), {'major': int(_maj), 'minor': int(_min), 'patch': int(_pat), 'suffix': ''})()\n"
            "version = get_runtime_version_info()\n"
            "def ValidateProtobufRuntimeVersion(*args, **kwargs):\n"
            "    return None\n"
        )
    print("PB_FILE", _pb_pkg / "__init__.py")
    print("PB_VER", _gp.__version__)
    print("RV_EXISTS_AFTER_SHIM", _rv.exists())
    try:
        import mediapipe as _mp
        print("MP_OK_TOP", _mp.__version__)
    except Exception as _e:
        print("MP_FAIL_TOP", repr(_e)[:400])
        raise SystemExit(1) from _e

import mpmath

OUT = Path("/kaggle/working") if ON_KAGGLE else Path("/tmp/daisee_gt_out")
OUT.mkdir(parents=True, exist_ok=True)
METRICS = OUT / "daisee_gt_metrics.json"
CACHE = OUT / "daisee_clips_v1.npz"            # per-clip stream tensors (reused)
RUNS = OUT / "daisee_gt_grid"

FAST = os.environ.get("GRID_FAST", "") == "1"
SEQ_K = 6                                        # 1-D Tchebichef order across time
FACE_SIZE = 64
FACE_K = 16                                      # 2-D Tchebichef order -> 16x16
N_CLASSES = 4                                    # DAiSEE Engagement 0..3 (ordinal)
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

STREAMS = ("face", "facemesh", "head_pose")
FUSIONS = ("gate", "core")
SKIPS = (True, False)
MDROPS = (True, False)
LOSSES = ("coral", "hybrid", "soft")

SHUFFLE_TOL = 0.10
MAX_CLIPS = 400 if FAST else None

device = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", device, "| FAST:", FAST)

# %% [markdown]
# ## Basis + transforms (identical to gtao_grid / pipeline/tchebichef.py)

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


def pad_time(arr, k):
    if arr.shape[0] >= k:
        return arr
    if arr.shape[0] == 0:
        raise ValueError("empty time axis")
    return np.concatenate([arr, np.repeat(arr[-1:], k - arr.shape[0], axis=0)], axis=0)

# %% [markdown]
# ## EngageNet loader: crops -> per-clip stream tensors (cached npz)
#
# The 4.15 GB dataset exposes face crops under
# `images_engagednet/ImageData3/{Train,Validation,Test}/<class>/subject_<id>_..._frame_<f>.jpg`.
# Face stream = Tchebichef moments of the crop (same front-end as DIPSER).
# Face-mesh = MediaPipe 478-refined-landmark mesh; head-pose = solvePnP
# (pitch, yaw, roll). Extraction runs once and caches to `CACHE`.

# %%
SUBJECT_RE = re.compile(
    r"^subject_(\d+)_(.*)_vid_(\d+)_(\d+)_frame_(\d+)$")


def find_crops():
    """Return list of (jpg_path, split, subject, clip) from any Kaggle mount."""
    if not ON_KAGGLE:
        root = Path(os.environ.get("ENGAGENET_ROOT", ""))
        if root.is_dir():
            search = [root]
        else:
            return []
    else:
        search = list(Path("/kaggle/input").glob("*"))
    found = []
    for base in search:
        for jpg in base.rglob("*.jpg"):
            rel = str(jpg)
            split = None
            for s in ("Train", "Validation", "Test"):
                if f"/{s}/" in rel or f"\\{s}\\" in rel:
                    split = s
                    break
            m = SUBJECT_RE.match(jpg.stem)
            if split is None or m is None:
                continue
            subj = f"subject_{m.group(1)}"
            found.append((jpg, split, subj, jpg.stem))
    return sorted(found, key=lambda r: r[0].as_posix())


class MeshExtractor:
    def __init__(self):
        self.fm = None
        try:
            import mediapipe as mp
            self.mp = mp
            self.fm = mp.solutions.face_mesh.FaceMesh(
                static_image_mode=True, max_num_faces=1, refine_landmarks=True,
                min_detection_confidence=0.5)
        except Exception as e:  # pragma: no cover
            print("mediapipe unavailable:", repr(e))
            self.fm = None

    def detect(self, jpg):
        img = cv2.cvtColor(cv2.imread(str(jpg)), cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]
        mesh, pose = None, None
        for attempt in range(2):
            pad = 0 if attempt == 0 else int(0.15 * max(h, w))
            im = (cv2.copyMakeBorder(img, pad, pad, pad, pad,
                                     cv2.BORDER_CONSTANT, value=(255, 255, 255))
                  if pad else img)
            res = self.fm.process(im)
            if res.multi_face_landmarks:
                lm = res.multi_face_landmarks[0]
                pts = np.array([(l.x * im.shape[1], l.y * im.shape[0], l.z)
                                for l in lm.landmark], dtype=np.float32)
                if pad:
                    pts[:, :2] -= pad
                mesh = pts
                pose = self._head_pose(pts.astype(np.float64), w, h)
                break
        return mesh, pose

    @staticmethod
    def _head_pose(pts, w, h):
        obj = np.array([
            [0.0, 0.0, 0.0], [0.0, -63.6, -12.5],
            [-43.3, 32.7, -26.0], [43.3, 32.7, -26.0],
            [-28.9, -28.9, -24.1], [28.9, -28.9, -24.1]], dtype=np.float64)
        idx = [1, 152, 33, 263, 61, 291]
        try:
            img = np.array([[x, y] for x, y, _ in pts[idx]], dtype=np.float64)
            f = max(w, h)
            mtx = np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1]],
                           dtype=np.float64)
            ok, rvec, _ = cv2.solvePnP(obj, img, mtx, None,
                                       flags=cv2.SOLVEPNP_ITERATIVE)
            if not ok:
                return None
            R, _ = cv2.Rodrigues(rvec)
            sy = math.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
            pitch = math.atan2(-R[2, 1], R[2, 2])
            yaw = math.atan2(R[2, 0], sy)
            roll = math.atan2(-R[1, 0], R[0, 0])
            return np.array([math.degrees(pitch), math.degrees(yaw),
                             math.degrees(roll)], dtype=np.float32)
        except Exception:
            return None


def encode_clip(jpg, split, subj, clip, label, ext):
    img = np.asarray(Image.open(jpg).convert("L").resize((FACE_SIZE, FACE_SIZE)),
                     dtype=np.float32) / 255.0
    face = transform_2d(img, FACE_K).ravel().astype(np.float32)      # 256
    mesh, pose = None, None
    if ext is not None:
        mesh, pose = ext.detect(jpg)
    return {"key": f"{split}/{subj}/{clip}", "subject": subj, "split": split,
            "face": face, "facemesh": mesh, "head_pose": pose,
            "label": CLASS_ORDER[label]}


def load_index():
    """Encode every available crop once; cache clip-level tensors to npz.

    Clip stream vector = pad_time to SEQ_K then a 1-D temporal Tchebichef, i.e.
    identical stream dims to DIPSER: face 1536, facemesh 8604, head_pose 18.
    """
    if CACHE.exists():
        print("loading cached clip corpus:", CACHE)
        return np.load(CACHE, allow_pickle=True)

    crops = find_crops()
    if not crops:
        print("no EngageNet crops found — synthetic smoke fallback")
        return synth_corpus()

    label_of = {}
    for jpg, split, subj, clip in crops:
        try:
            label_of[(split, clip)] = jpg.parent.name
        except Exception:
            continue
    if not label_of:
        raise SystemExit("no labelled EngageNet crops")

    per_frame = collections.defaultdict(list)
    ext = None
    if cv2 is not None:
        ext = MeshExtractor()
    t0 = time.time()
    for n, (jpg, split, subj, clip) in enumerate(crops):
        if (split, clip) not in label_of:
            continue
        if MAX_CLIPS and len(per_frame) >= MAX_CLIPS:
            break
        e = encode_clip(jpg, split, subj, clip, label_of[(split, clip)], ext)
        per_frame[(split, clip)].append(e)
        if (n + 1) % 2000 == 0:
            print(f"  {n + 1}/{len(crops)} crops, {len(per_frame)} clips "
                  f"[{time.time() - t0:.0f}s]", flush=True)

    records = []
    for (split, clip), entries in per_frame.items():
        entries.sort(key=lambda e: int(re.search(r"_frame_(\d+)$",
                                                 e["key"]).group(1)))
        first = entries[0]
        label = first["label"]
        faces = np.stack([e["face"] for e in entries])                    # (T,256)
        feats = {"face": transform_1d(
            np.moveaxis(pad_time(faces, SEQ_K), 0, -1).reshape(-1, SEQ_K),
            SEQ_K).ravel()}
        for s_key, stackf in (("facemesh", lambda: np.stack(
                [e for e in (x["facemesh"] for x in entries) if e is not None])),
                              ("head_pose", lambda: np.stack(
                [e for e in (x["head_pose"] for x in entries) if e is not None]))):
            arr = None
            try:
                arr = stackf()
            except Exception:
                arr = None
            if arr is not None and arr.shape[0] > 0:
                feats[s_key] = transform_1d(
                    np.moveaxis(pad_time(arr, SEQ_K), 0, -1)
                    .reshape(-1, SEQ_K), SEQ_K).ravel()
        soft = np.zeros(N_CLASSES, dtype=np.float32)
        soft[label] = 1.0
        records.append({"key": first["key"], "subject": first["subject"],
                        "split": first["split"], "label": label, "soft": soft,
                        **feats})

    keys = np.asarray([r["key"] for r in records])
    subjects = np.asarray([r["subject"] for r in records])
    splits = np.asarray([r["split"] for r in records])
    y = np.asarray([r["label"] for r in records], dtype=np.int64)
    soft = np.stack([r["soft"] for r in records]).astype(np.float32)
    avail = {s: np.asarray([r.get(s) is not None for r in records], dtype=bool)
             for s in STREAMS}
    feats = {s: None for s in STREAMS}
    for s in STREAMS:
        idx = [i for i, r in enumerate(records) if r.get(s) is not None]
        if idx:
            arr = np.asarray([records[i][s] for i in idx], dtype=np.float32)
            full = np.full((len(records), arr.shape[1]), np.nan, dtype=np.float32)
            full[idx] = arr
            feats[s] = full
        print(f"stream {s}: present in {len(idx)}/{len(records)} "
              f"({100 * len(idx) / (len(records) or 1):.1f}%)")
    empty2d = np.zeros((0, 0), dtype=np.float32)
    np.savez(CACHE, keys=keys, subjects=subjects, splits=splits, y=y, soft=soft,
             avail_face=avail["face"], avail_facemesh=avail["facemesh"],
             avail_head_pose=avail["head_pose"],
             feat_face=feats["face"] if feats["face"] is not None else empty2d,
             feat_facemesh=feats["facemesh"] if feats["facemesh"] is not None else empty2d,
             feat_head_pose=feats["head_pose"] if feats["head_pose"] is not None else empty2d)
    print(f"corpus: {len(records)} clips, {len(set(subjects.tolist()))} subjects, "
          f"{time.time() - t0:.0f}s")
    return np.load(CACHE, allow_pickle=True)


def synth_corpus():
    """Local GRID_FAST smoke: class-dependent per-subject random faces."""
    rng = np.random.default_rng(0)
    n = 200
    records = []
    val_subjects = {f"subject_{i}" for i in range(4)}
    for i in range(n):
        subj = f"subject_{i % 20}"
        split = "Validation" if subj in val_subjects else "Train"
        label = int(i % N_CLASSES)
        face = rng.normal(size=(1536,)).astype(np.float32)
        face += 0.1 * label
        mesh = rng.normal(size=(8604,)).astype(np.float32) + 0.05 * label
        hp = rng.normal(size=(18,)).astype(np.float32) + 0.2 * label
        soft = np.zeros(N_CLASSES, dtype=np.float32)
        soft[label] = 1.0
        records.append({"key": f"{split}/{subj}/{i}", "subject": subj,
                        "split": split, "label": label, "soft": soft,
                        "face": face, "facemesh": mesh, "head_pose": hp})
    keys = np.asarray([r["key"] for r in records])
    subjects = np.asarray([r["subject"] for r in records])
    splits = np.asarray([r["split"] for r in records])
    y = np.asarray([r["label"] for r in records], dtype=np.int64)
    soft = np.stack([r["soft"] for r in records]).astype(np.float32)
    np.savez(CACHE, keys=keys, subjects=subjects, splits=splits, y=y, soft=soft,
             avail_face=np.ones(n, dtype=bool), avail_facemesh=np.ones(n, dtype=bool),
             avail_head_pose=np.ones(n, dtype=bool),
             feat_face=np.stack([r["face"] for r in records]),
             feat_facemesh=np.stack([r["facemesh"] for r in records]),
             feat_head_pose=np.stack([r["head_pose"] for r in records]))
    return np.load(CACHE, allow_pickle=True)

# %% [markdown]
# ## DAiSEE loader: .avi clips -> per-clip stream tensors (cached npz)
#
# Front-end identical to EngageNet/DIPSER: per sampled frame run MediaPipe
# FaceMesh (478 refined landmarks) -> face crop (Tchebichef moments) + landmark
# mesh + solvePnP head pose. Expected layout (verified for olgaparfenova/daisee):
#   `**/Labels/{Train,Validation,Test}Labels.csv`  (ClipID, Boredom, Engagement, ...)
#   `**/DataSet/{Train,Validation,Test}/<userID>/.../<clip>.avi`
# The userID folder is the subject group; split comes from the labels CSV.

# %%
def read_daisee_labels():
    """clip_id (no ext) -> (engagement 0..3, split) across all *Labels.csv."""
    labels = {}
    roots = [Path("/kaggle/input")] if ON_KAGGLE else [
        Path(os.environ.get("DAISEE_ROOT", ""))]
    found = 0
    for root in roots:
        if not root.is_dir():
            continue
        for f in sorted(root.rglob("*Labels.csv")):
            split = ("Train" if "train" in f.name.lower() else
                     "Validation" if "valid" in f.name.lower() else
                     "Test" if "test" in f.name.lower() else None)
            if split is None:
                continue
            found += 1
            try:
                with open(f, newline="") as fh:
                    rd = csv.DictReader(fh)
                    cols = {c.lower().strip(): c for c in (rd.fieldnames or [])}
                    cid_col = cols.get("clipid") or cols.get("clip id")
                    eng_col = cols.get("engagement")
                    if not cid_col or not eng_col:
                        print("  skip labels (no clipid/engagement col):", f.name)
                        continue
                    for row in rd:
                        try:
                            cid = Path(str(row[cid_col])).stem
                            labels[cid] = (int(float(row[eng_col])), split)
                        except Exception:
                            continue
            except Exception as e:
                print("  label read skip", f, repr(e))
    print(f"labels: {len(labels)} clips from {found} CSV(s)")
    return labels


def find_videos():
    if ON_KAGGLE:
        roots = [Path("/kaggle/input")]
    else:
        roots = [Path(os.environ.get("DAISEE_ROOT", ""))]
    out = []
    for root in roots:
        if root.is_dir():
            out += sorted(root.rglob("*.avi"))
    return out


def subject_of(avi):
    parts = avi.parts
    cand = parts[-3].lower()
    if cand in ("train", "validation", "test"):
        return parts[-2]
    return parts[-3]


class MeshExtractor:
    """MediaPipe FaceMesh (478, refine_landmarks) + solvePnP head pose + face bbox."""

    def __init__(self):
        self.fm = None
        try:
            import mediapipe as mp
            self.mp = mp
            self.fm = mp.solutions.face_mesh.FaceMesh(
                static_image_mode=True, max_num_faces=1, refine_landmarks=True,
                min_detection_confidence=0.5)
        except Exception as e:  # pragma: no cover
            print("mediapipe unavailable:", repr(e))
            self.fm = None

    def detect(self, rgb):
        """-> (mesh(478,3) or None, pose(3,) or None, face_bbox(x0,y0,x1,y1))"""
        if self.fm is None:
            return None, None, None
        h, w = rgb.shape[:2]
        for attempt in range(2):
            pad = 0 if attempt == 0 else int(0.1 * max(h, w))
            im = (cv2.copyMakeBorder(rgb, pad, pad, pad, pad,
                                     cv2.BORDER_CONSTANT, value=(255, 255, 255))
                  if pad else rgb)
            res = self.fm.process(im)
            if res.multi_face_landmarks:
                lm = res.multi_face_landmarks[0]
                pts = np.array([(l.x * im.shape[1], l.y * im.shape[0], l.z)
                                for l in lm.landmark], dtype=np.float32)
                if pad:
                    pts[:, :2] -= pad
                pose = self._head_pose(pts.astype(np.float64), w, h)
                xs, ys = pts[:, 0], pts[:, 1]
                m = 0.15
                x0 = max(0, int(xs.min() - m * (xs.max() - xs.min())))
                y0 = max(0, int(ys.min() - m * (ys.max() - ys.min())))
                x1 = min(w - 1, int(xs.max() + m * (xs.max() - xs.min())))
                y1 = min(h - 1, int(ys.max() + m * (ys.max() - ys.min())))
                return pts, pose, (x0, y0, x1, y1)
        return None, None, None

    @staticmethod
    def _head_pose(pts, w, h):
        obj = np.array([
            [0.0, 0.0, 0.0], [0.0, -63.6, -12.5],
            [-43.3, 32.7, -26.0], [43.3, 32.7, -26.0],
            [-28.9, -28.9, -24.1], [28.9, -28.9, -24.1]], dtype=np.float64)
        idx = [1, 152, 33, 263, 61, 291]
        try:
            img = np.array([[x, y] for x, y, _ in pts[idx]], dtype=np.float64)
            f = max(w, h)
            mtx = np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1]],
                           dtype=np.float64)
            ok, rvec, _ = cv2.solvePnP(obj, img, mtx, None,
                                       flags=cv2.SOLVEPNP_ITERATIVE)
            if not ok:
                return None
            R, _ = cv2.Rodrigues(rvec)
            sy = math.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
            pitch = math.atan2(-R[2, 1], R[2, 2])
            yaw = math.atan2(R[2, 0], sy)
            roll = math.atan2(-R[1, 0], R[0, 0])
            return np.array([math.degrees(pitch), math.degrees(yaw),
                             math.degrees(roll)], dtype=np.float32)
        except Exception:
            return None


def frame_streams(frame_bgr, ext):
    """Per-frame -> face moment vector, optional mesh/pose (raw)."""
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    mesh, pose, bbox = (ext.detect(rgb) if ext is not None else
                        (None, None, None))
    if bbox is None:
        h, w = rgb.shape[:2]
        side = int(0.8 * min(h, w))
        y0 = (h - side) // 2
        x0 = (w - side) // 2
        bbox = (x0, y0, x0 + side, y0 + side)
    x0, y0, x1, y1 = bbox
    crop = cv2.cvtColor(rgb[y0:y1, x0:x1], cv2.COLOR_RGB2GRAY)
    crop = cv2.resize(crop, (FACE_SIZE, FACE_SIZE)).astype(np.float32) / 255.0
    face = transform_2d(crop, FACE_K).ravel().astype(np.float32)
    return {"face": face, "facemesh": mesh, "head_pose": pose}


def encode_clip(video, clip_id, label, split, subject, ext):
    cap = cv2.VideoCapture(str(video))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    if total <= 0:
        cap.release()
        return None
    idxs = np.linspace(0, total - 1, SEQ_K).astype(int)
    frames = []
    for i in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, fr = cap.read()
        if ok:
            frames.append(fr)
    cap.release()
    if not frames:
        return None
    per = [frame_streams(f, ext) for f in frames]

    faces = np.stack([p["face"] for p in per])
    feats = {"face": transform_1d(
        np.moveaxis(pad_time(faces, SEQ_K), 0, -1).reshape(-1, SEQ_K),
        SEQ_K).ravel()}
    for s_key in ("facemesh", "head_pose"):
        arrs = [p[s_key] for p in per if p[s_key] is not None]
        if not arrs:
            continue
        arr = np.stack(arrs)
        feats[s_key] = transform_1d(
            np.moveaxis(pad_time(arr, SEQ_K), 0, -1).reshape(-1, SEQ_K),
            SEQ_K).ravel()
    soft = np.zeros(N_CLASSES, dtype=np.float32)
    soft[label] = 1.0
    return {"key": f"{split}/{subject}/{clip_id}", "subject": subject,
            "split": split, "label": label, "soft": soft, **feats}


def load_daisee():
    """Encode every labelled clip once; cache clip-level tensors to npz."""
    if CACHE.exists():
        print("loading cached clip corpus:", CACHE)
        return np.load(CACHE, allow_pickle=True)

    labels = read_daisee_labels()
    videos = find_videos()
    if not videos:
        print("no DAiSEE videos found — synthetic smoke fallback")
        return synth_corpus()
    if not labels:
        raise SystemExit("no DAiSEE labels found")

    mask = np.asarray([v.stem in labels for v in videos])
    labelled = [v for v, ok in zip(videos, mask) if ok]
    print(f"videos: {len(videos)} total, {len(labelled)} labelled "
          f"({100 * mask.mean():.1f}%)")

    ext = MeshExtractor() if cv2 is not None else None
    if ext is not None and ext.fm is None:
        ext = None
    records = []
    t0 = time.time()
    for n, video in enumerate(labelled):
        if MAX_CLIPS and len(records) >= MAX_CLIPS:
            break
        clip_id = video.stem
        label, split = labels[clip_id]
        subject = subject_of(video)
        r = encode_clip(video, clip_id, label, split, subject, ext)
        if r is not None:
            records.append(r)
        if (n + 1) % 1000 == 0:
            print(f"  {n + 1}/{len(labelled)} clips encoded "
                  f"[{time.time() - t0:.0f}s]", flush=True)

    if not records:
        raise SystemExit("no clips encoded successfully")
    keys = np.asarray([r["key"] for r in records])
    subjects = np.asarray([r["subject"] for r in records])
    splits = np.asarray([r["split"] for r in records])
    y = np.asarray([r["label"] for r in records], dtype=np.int64)
    soft = np.stack([r["soft"] for r in records]).astype(np.float32)
    avail = {s: np.asarray([r.get(s) is not None for r in records], dtype=bool)
             for s in STREAMS}
    feats = {s: None for s in STREAMS}
    for s in STREAMS:
        idx = [i for i, r in enumerate(records) if r.get(s) is not None]
        if idx:
            arr = np.asarray([records[i][s] for i in idx], dtype=np.float32)
            full = np.full((len(records), arr.shape[1]), np.nan, dtype=np.float32)
            full[idx] = arr
            feats[s] = full
        print(f"stream {s}: present in {len(idx)}/{len(records)} "
              f"({100 * len(idx) / (len(records) or 1):.1f}%)")
    empty2d = np.zeros((0, 0), dtype=np.float32)
    np.savez(CACHE, keys=keys, subjects=subjects, splits=splits, y=y, soft=soft,
             avail_face=avail["face"], avail_facemesh=avail["facemesh"],
             avail_head_pose=avail["head_pose"],
             feat_face=feats["face"] if feats["face"] is not None else empty2d,
             feat_facemesh=feats["facemesh"] if feats["facemesh"] is not None else empty2d,
             feat_head_pose=feats["head_pose"] if feats["head_pose"] is not None else empty2d)
    print(f"corpus: {len(records)} clips, {len(set(subjects.tolist()))} subjects, "
          f"{time.time() - t0:.0f}s")
    return np.load(CACHE, allow_pickle=True)


def synth_corpus():
    """Local GRID_FAST smoke: class-dependent per-subject random faces."""
    rng = np.random.default_rng(0)
    n = 200
    records = []
    val_subjects = {f"subject_{i}" for i in range(4)}
    for i in range(n):
        subj = f"subject_{i % 20}"
        split = "Validation" if subj in val_subjects else "Train"
        label = int(i % N_CLASSES)
        face = rng.normal(size=(1536,)).astype(np.float32)
        face += 0.1 * label
        mesh = rng.normal(size=(8604,)).astype(np.float32) + 0.05 * label
        hp = rng.normal(size=(18,)).astype(np.float32) + 0.2 * label
        soft = np.zeros(N_CLASSES, dtype=np.float32)
        soft[label] = 1.0
        records.append({"key": f"{split}/{subj}/{i}", "subject": subj,
                        "split": split, "label": label, "soft": soft,
                        "face": face, "facemesh": mesh, "head_pose": hp})
    keys = np.asarray([r["key"] for r in records])
    subjects = np.asarray([r["subject"] for r in records])
    splits = np.asarray([r["split"] for r in records])
    y = np.asarray([r["label"] for r in records], dtype=np.int64)
    soft = np.stack([r["soft"] for r in records]).astype(np.float32)
    np.savez(CACHE, keys=keys, subjects=subjects, splits=splits, y=y, soft=soft,
             avail_face=np.ones(n, dtype=bool),
             avail_facemesh=np.ones(n, dtype=bool),
             avail_head_pose=np.ones(n, dtype=bool),
             feat_face=np.stack([r["face"] for r in records]),
             feat_facemesh=np.stack([r["facemesh"] for r in records]),
             feat_head_pose=np.stack([r["head_pose"] for r in records]))
    return np.load(CACHE, allow_pickle=True)
# %% [markdown]
# ## Grid model (copy of gtao_grid.py: gate / CORE fusion, skip, mdrop, CORAL)

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
# ## Metrics + training helpers (gtao_grid.py)

# %%
import collections  # noqa: E402
import socket      # noqa: E402
import sklearn      # noqa: E402
from sklearn.metrics import cohen_kappa_score  # noqa: E402


def qwk(y_true, y_pred):
    if len(y_true) < 2 or len(np.unique(y_true)) < 2:
        return 0.0
    try:
        v = float(cohen_kappa_score(y_true, y_pred, weights="quadratic"))
    except ValueError:
        v = 0.0
    return 0.0 if not math.isfinite(v) else v


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


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)


def fit_fold(model, Xtr, ytr, Xva, yva, stream_list, loss="coral", softtr=None):
    set_seed(int(np.random.randint(2 ** 31)))
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
                    "state": {k: v.detach().clone()
                              for k, v in model.state_dict().items()},
                    "epoch": epoch}
        if epoch % 5 == 0:
            with torch.no_grad():
                _, alpha = model({k: Xva[k] for k in stream_list}, apply_mdrop=False)
            gate = ([round(float(a), 4) for a in alpha.mean(dim=0).cpu().numpy()]
                    if alpha is not None else [])
            trace.append({"epoch": epoch, "val_qwk": va["qwk"], "gate": gate})
        if epoch >= MIN_EPOCHS and epoch - best["epoch"] >= PATIENCE:
            break
    model.load_state_dict(best["state"])
    model.eval()
    return best, trace, {"epochs": epoch, "loss_last": round(float(np.mean(losses[-64:])), 4)}


def cfg_name(cfg):
    if cfg.get("unimodal"):
        return f"unimodal_{cfg['unimodal']}"
    if cfg.get("shuffle"):
        return "shuffle_core_skip1_drop1"
    base = f"{cfg['fusion']}_skip{int(cfg['use_skip'])}_drop{int(cfg['mdrop'])}"
    return base if cfg.get("loss", "coral") == "coral" else base + f"_{cfg['loss']}"

# %% [markdown]
# ## Grid drivers
#
# `run_protocol(corpus, folds_by_seed, cfg, seed)` trains `cfg` across the given
# `(train, val)` folds for one seed and pools OOF predictions. `folds_by_seed`
# maps seed → list of `(idx_train, idx_val)` pairs (P2: GroupKFold-5; P1: one
# subject-filtered train/val pair).

# %%
def run_protocol(corpus, folds, cfg, seed):
    name = cfg_name(cfg)
    ys = corpus["y"]
    soft_full = corpus["soft"]
    stream_list = ([cfg["unimodal"]] if cfg.get("unimodal") else
                   [s for s in STREAMS if corpus[f"feat_{s}"].shape[0] > 0])
    dims = {s: int(corpus[f"feat_{s}"].shape[1]) for s in stream_list
            if corpus[f"feat_{s}"].shape[0] > 0}
    if not dims:
        return None

    idx_all = np.arange(len(ys))
    y_run = ys.copy()
    soft_run = soft_full.copy()
    if cfg.get("shuffle"):
        rng = np.random.default_rng(seed)
        perm = rng.permutation(len(idx_all))
        y_run[idx_all] = ys[idx_all][perm]
        soft_run[idx_all] = soft_full[idx_all][perm]
        print("shuffle control: labels permuted within whole population")

    oof_pred = np.full(len(ys), np.nan)
    meta = {"probe": None, "folds": [], "gate_trace": {}}
    for f, (tr, te) in enumerate(folds):
        if len(set(y_run[tr])) < 2 or not te.any():
            continue
        Xtr = to_torch({s: corpus[f"feat_{s}"][tr] for s in stream_list})
        ytr = torch.as_tensor(y_run[tr], device=device)
        softtr = (torch.as_tensor(soft_run[tr], device=device)
                  if cfg.get("loss") == "soft" else None)
        Xva = to_torch({s: corpus[f"feat_{s}"][te] for s in stream_list})
        yva = torch.as_tensor(y_run[te], device=device)
        set_seed(int(f"{seed}{f}"))
        if cfg.get("unimodal"):
            model = GTAONetGrid(dims, fusion="core", use_skip=False,
                                mdrop=False).to(device)
        else:
            model = GTAONetGrid(dims, fusion=cfg["fusion"], use_skip=cfg["use_skip"],
                                mdrop=cfg["mdrop"]).to(device)
        t0 = time.time()
        best, trace, metas = fit_fold(model, Xtr, ytr, Xva, yva, stream_list,
                                      loss=cfg.get("loss", "coral"), softtr=softtr)
        with torch.no_grad():
            logits, _ = model(Xva, apply_mdrop=False)
            oof_pred[te] = model.head.predict(logits).cpu().numpy()
        meta["gate_trace"][f] = trace
        meta["folds"].append({"fold": f, "n_train": int(len(tr)),
                              "n_val": int(len(te)), "best_val_qwk": best["qwk"],
                              "best_epoch": best["epoch"], **metas,
                              "wall_s": round(time.time() - t0, 1)})
        print(f"  fold {f}: val n={len(te)} QWK={best['qwk']:.4f} "
              f"ep={best['epoch']}  [{round(time.time() - t0, 1)}s]", flush=True)

    ok = np.isfinite(oof_pred)
    meta["oof"] = {"keys": corpus["keys"][ok].astype(str).tolist(),
                   "y": ys[ok].astype(int).tolist(),
                   "pred": oof_pred[ok].astype(int).tolist(),
                   "subject": corpus["subjects"][ok].astype(str).tolist(),
                   "split": corpus["splits"][ok].astype(str).tolist()}
    meta["pooled"] = {"qwk": round(qwk(ys[ok].astype(int),
                                       oof_pred[ok].astype(int)), 4),
                      "acc": round(float(np.mean(oof_pred[ok].astype(int)
                                                 == ys[ok].astype(int))), 4),
                      "mae": round(float(np.mean(np.abs(oof_pred[ok].astype(int)
                                                        - ys[ok].astype(int)))), 4),
                      "n": int(ok.sum()),
                      "n_subjects": len(set(corpus["subjects"][ok].tolist()))}
    return meta


def subject_group_folds(corpus, seed, n_folds=N_FOLDS):
    uniq = sorted(set(corpus["subjects"].tolist()))
    rng = np.random.default_rng(seed)
    rng.shuffle(uniq)
    fold_of_subj = {s: i % n_folds for i, s in enumerate(uniq)}
    fold_of = np.array([fold_of_subj[s] for s in corpus["subjects"]])
    return [(np.flatnonzero(fold_of != f), np.flatnonzero(fold_of == f))
            for f in range(n_folds)]


def official_split_folds(corpus):
    """P1: train on Train minus subjects seen in Validation; eval on Validation."""
    splits = corpus["splits"]
    tr_mask = splits == "Train"
    va_mask = splits == "Validation"
    leak_subjects = set(corpus["subjects"][va_mask].tolist())
    clean_tr = tr_mask & np.array([s not in leak_subjects
                                   for s in corpus["subjects"]])
    return [(np.flatnonzero(clean_tr), np.flatnonzero(va_mask))]


def summarize(configs, doc_key, run_dir, write_to):
    rows = []
    for cfg in configs:
        name = cfg_name(cfg)
        d = run_dir / name
        qwks, accs, maes, ns, probes, gates = [], [], [], [], [], []
        for seed in SEEDS:
            f = d / f"seed{seed}" / "oof.json"
            if not f.exists():
                continue
            m = json.loads(f.read_text())
            if not m.get("pooled"):
                continue
            p = m["pooled"]
            qwks.append(p["qwk"]); accs.append(p["acc"]); maes.append(p["mae"])
            ns.append(p["n"]); probes.append(m.get("probe"))
            gates.append(m.get("gate_trace"))
        if not qwks:
            continue
        rows.append({
            "config": name, "fusion": ("unimodal" if cfg.get("unimodal")
                                       else cfg["fusion"]),
            "skip": cfg.get("use_skip"), "mdrop": cfg.get("mdrop"),
            "loss": cfg.get("loss", "coral"), "unimodal": cfg.get("unimodal"),
            "shuffle": cfg.get("shuffle"),
            "qwk_mean": round(float(np.mean(qwks)), 4),
            "qwk_std": round(float(np.std(qwks)), 4),
            "qwk_per_seed": [round(q, 4) for q in qwks],
            "acc_mean": round(float(np.mean(accs)), 4),
            "mae_mean": round(float(np.mean(maes)), 4),
            "n_median": int(sorted(ns)[len(ns) // 2]),
            "probe_seed0": probes[0] if probes and probes[0] else None})
        if gates and gates[0]:
            tr = sorted(gates[0].items())[0][1]
            if tr:
                rows[-1]["gate_alpha_final"] = tr[-1]["gate"]
    rows.sort(key=lambda r: (-r["qwk_mean"], r["config"]))
    print(f"\n=== {doc_key} (pooled {('over 5 folds' if 'P2' in doc_key else 'train/val')}"
          f", seeds {list(SEEDS)}) ===")
    print(f"{'config':28s} {'QWK mean±std':>14s} {'acc':>7s} {'mae':>6s} {'n':>5s}")
    for r in rows:
        print(f"{r['config']:28s} {r['qwk_mean']:+.4f}±{r['qwk_std']:.3f} "
              f"{r['acc_mean']:7.3f} {r['mae_mean']:6.3f} {r['n_median']:5d}")

    shuffle_rows = [r for r in rows if r.get("shuffle")]
    if shuffle_rows:
        mx = max(abs(r["qwk_mean"]) for r in shuffle_rows)
        print(f"shuffle control: max |QWK| = {mx:.4f} "
              f"({'PASS' if mx <= SHUFFLE_TOL else 'FAIL'} ≤ {SHUFFLE_TOL})")
    uni_rows = [r for r in rows if r.get("unimodal")]
    if uni_rows:
        best = max(uni_rows, key=lambda r: r["qwk_mean"])
        print(f"best in-protocol unimodal: {best['config']} = {best['qwk_mean']:.4f}")

    doc = json.loads(write_to.read_text()) if write_to.exists() else {}
    doc.setdefault(doc_key, {})["rows"] = rows
    write_to.write_text(json.dumps(doc, indent=2))
    return rows


def masked_eval(corpus, seeds=SEEDS):
    """H4: retrain gate/core skip1-drop0-soft; zero out each stream at eval."""
    configs = ({'fusion': 'gate', 'use_skip': True, 'mdrop': False, 'loss': 'soft'},
               {'fusion': 'core', 'use_skip': True, 'mdrop': False, 'loss': 'soft'})
    all_present = [s for s in STREAMS if corpus[f"feat_{s}"].shape[0] > 0]
    keep = np.ones(len(corpus["y"]), dtype=bool)
    for s in all_present:
        keep &= corpus[f"avail_{s}"]
    idx_all = np.flatnonzero(keep)
    ys = corpus["y"]
    per_seed = {}
    for seed in seeds:
        uniq = sorted(set(corpus["subjects"][idx_all].tolist()))
        rng = np.random.default_rng(seed)
        rng.shuffle(uniq)
        fold_of_subj = {s: i % N_FOLDS for i, s in enumerate(uniq)}
        fold_of = np.array([fold_of_subj[s] for s in corpus["subjects"][idx_all]])
        rows = {}
        for cfg in configs:
            name = cfg_name(cfg)
            cond_pred = {c: np.full(len(corpus["y"]), np.nan)
                         for c in ["none"] + [f"mask_{s}" for s in all_present]}
            alpha_acc = {c: [] for c in cond_pred}
            dims = {s: int(corpus[f"feat_{s}"].shape[1]) for s in all_present}
            for f in range(N_FOLDS):
                te = idx_all[fold_of == f]
                tr = idx_all[~np.isin(idx_all, te)]
                if len(set(ys[tr])) < 2 or len(te) == 0:
                    continue
                Xtr = to_torch({s: corpus[f"feat_{s}"][tr] for s in all_present})
                ytr = torch.as_tensor(ys[tr], device=device)
                softtr = torch.as_tensor(corpus["soft"][tr], device=device)
                Xva = to_torch({s: corpus[f"feat_{s}"][te] for s in all_present})
                yva = torch.as_tensor(ys[te], device=device)
                set_seed(int(f"{seed}{f}"))
                model = GTAONetGrid(dims, fusion=cfg["fusion"],
                                    use_skip=cfg["use_skip"], mdrop=cfg["mdrop"])
                model = model.to(device)
                best, _, _ = fit_fold(model, Xtr, ytr, Xva, yva, all_present,
                                      loss=cfg["loss"], softtr=softtr)
                model.load_state_dict(best["state"]); model.eval()
                with torch.no_grad():
                    logits, alpha = model(Xva, apply_mdrop=False)
                    cond_pred["none"][te] = model.head.predict(logits).cpu().numpy()
                    if alpha is not None:
                        alpha_acc["none"].append(alpha.cpu().numpy())
                    for s in all_present:
                        Xm = {k: (torch.zeros_like(v) if k == s else v)
                              for k, v in Xva.items()}
                        logits_m, alpha_m = model(Xm, apply_mdrop=False)
                        cond_pred[f"mask_{s}"][te] = (
                            model.head.predict(logits_m).cpu().numpy())
                        if alpha_m is not None:
                            alpha_acc[f"mask_{s}"].append(alpha_m.cpu().numpy())
            out = {"config": cfg, "seed": seed, "conditions": {}}
            for c, pred in cond_pred.items():
                ok = np.isfinite(pred)
                entry = {"qwk": round(qwk(ys[ok].astype(int),
                                          pred[ok].astype(int)), 4),
                         "acc": round(float(np.mean(pred[ok].astype(int)
                                                    == ys[ok].astype(int))), 4),
                         "mae": round(float(np.mean(np.abs(
                             pred[ok].astype(int) - ys[ok].astype(int)))), 4),
                         "n": int(ok.sum())}
                if alpha_acc[c]:
                    a = np.concatenate(alpha_acc[c], axis=0)
                    entry["gate_alpha"] = [round(float(x), 4)
                                           for x in a.mean(axis=0)]
                out["conditions"][c] = entry
            rows[name] = out
            print(f"  H4 {name} seed {seed}: "
                  f"none {out['conditions']['none']['qwk']:.3f} -> "
                  f"max masked drop "
                  f"{max(out['conditions']['none']['qwk'] - v['qwk'] for k, v in out['conditions'].items() if k.startswith('mask_')):.3f}")
        per_seed[seed] = rows
    return per_seed

# %% [markdown]
# ## Protocol P1 (official split, subject-filtered) — direct SOTA comparison

# %%
def run_p1(corpus):
    run_dir = RUNS / "P1_official"
    folds = official_split_folds(corpus)
    tr, te = folds[0]
    print(f"P1 | train n={len(tr)}  val n={len(te)}  "
          f"train subjects={len(set(corpus['subjects'][tr].tolist()))}  "
          f"val subjects={len(set(corpus['subjects'][te].tolist()))}")
    configs = [dict(z) for z in ({"fusion": f, "use_skip": sk, "mdrop": md,
                                  "loss": lo}
                                 for f, sk, md, lo in
                                 product(FUSIONS, SKIPS, MDROPS, LOSSES))]
    for s in STREAMS:
        configs.append({"unimodal": s, "loss": "coral"})
    for cfg in configs:
        for seed in SEEDS:
            f = run_dir / cfg_name(cfg) / f"seed{seed}" / "oof.json"
            if f.exists():
                continue
            m = run_protocol(corpus, folds, cfg, seed)
            if m is None:
                continue
            f.parent.mkdir(parents=True, exist_ok=True)
            f.write_text(json.dumps(m, indent=1))
    rows = summarize(configs, "protocol_P1_official", run_dir, METRICS)

# %% [markdown]
# ## Protocol P2 (GroupKFold-5 by subject, full factorial) — DIPSER-style

# %%
def run_p2(corpus):
    run_dir = RUNS / "P2_groupkfold"
    configs = []
    for f, sk, md, lo in product(FUSIONS, SKIPS, MDROPS, LOSSES):
        configs.append({"fusion": f, "use_skip": sk, "mdrop": md, "loss": lo})
    for s in STREAMS:
        configs.append({"unimodal": s, "loss": "coral"})
    configs.append({"shuffle": True, "fusion": "core", "use_skip": True,
                    "mdrop": True, "loss": "coral"})
    for seed in SEEDS:
        folds = subject_group_folds(corpus, seed)
        nd = np.bincount(corpus["y"])
        print(f"P2 seed {seed} | n={len(corpus['y'])} "
              f"subjects={len(set(corpus['subjects'].tolist()))} "
              f"class_dist={nd.tolist()}")
        for cfg in configs:
            name = cfg_name(cfg)
            f = run_dir / name / f"seed{seed}" / "oof.json"
            if f.exists():
                continue
            m = run_protocol(corpus, folds, cfg, seed)
            if m is None:
                continue
            f.parent.mkdir(parents=True, exist_ok=True)
            f.write_text(json.dumps(m, indent=1))
    rows = summarize(configs, "protocol_P2_groupkfold", run_dir, METRICS)
    h4 = masked_eval(corpus, seeds=SEEDS)
    h4_key = "h4_masked_eval_P2"
    doc = json.loads(METRICS.read_text()) if METRICS.exists() else {}
    doc.setdefault(h4_key, {})["per_seed"] = {
        str(s): {n: {c: out["conditions"][c]
                     for c in out["conditions"]}
                 for n, out in gr.items()}
        for s, gr in h4.items()}
    METRICS.write_text(json.dumps(doc, indent=2))
    return rows

# %% [markdown]
# ## Main

# %%
def analyze(corpus):
    splits = corpus["splits"]
    subs = corpus["subjects"]
    print("\n== population ==")
    for sp in ("Train", "Validation", "Test"):
        mask = splits == sp
        print(f"  {sp:10s} clips={int(mask.sum()):5d} "
              f"subjects={len(set(subs[mask].tolist())):3d} "
              f"class={np.bincount(corpus['y'][mask], minlength=N_CLASSES).tolist()}")
    ov = {a + "|" + b: len(set(subs[splits == a].tolist())
                            & set(subs[splits == b].tolist()))
          for a, b in [("Train", "Validation"), ("Train", "Test"),
                       ("Validation", "Test")]}
    print(f"  subject overlap (Kaggle splits, NOT subject-disjoint): {ov}")


def main():
    t0 = time.time()
    corpus = load_daisee()
    METRICS.parent.mkdir(parents=True, exist_ok=True)
    doc = json.loads(METRICS.read_text()) if METRICS.exists() else {}
    doc["status"] = "running"
    doc["env"] = {"device": device, "fast": FAST, "host": socket.gethostname()}
    METRICS.write_text(json.dumps(doc, indent=2))
    analyze(corpus)
    run_p1(corpus)
    run_p2(corpus)
    doc = json.loads(METRICS.read_text())
    doc["status"] = "done"
    doc["finished_utc"] = time.strftime("%Y-%m-%d %H:%M:%S")
    doc["wall_s"] = round(time.time() - t0, 1)
    METRICS.write_text(json.dumps(doc, indent=2))
    print("total wall:", round(time.time() - t0, 1), "s")


if __name__ == "__main__":
    main()