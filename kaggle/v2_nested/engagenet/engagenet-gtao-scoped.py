# === v2 NESTED: derived from v1 by tools/make_nested.py. Early stopping now uses an
# === inner subject split of the training fold (GTAO_NESTED=0 restores v1 behaviour).
"""gTAO-Net on EngageNet: a scoped 3-stream engagement benchmark (v5, v2-nested).

WHAT THIS SCRIPT DOES
---------------------
It trains a small neural network, "gTAO-Net", that predicts how engaged a
person is in a short video clip. The label has 4 ordered levels:

    Not-Engaged (0) < Barely-Engaged (1) < Engaged (2) < Highly-Engaged (3)

Each clip is described by three "streams" of numbers (feature vectors):

    * face       - 2-D Tchebichef moments of the grey face crop   -> 1536 numbers
    * facemesh   - MediaPipe 478 x 3 face landmarks per frame     -> 8604 numbers
    * head_pose  - pitch / yaw / roll of the head per frame       ->   18 numbers

The time axis (6 frames, SEQ_K = 6) is compressed with a 1-D Tchebichef
transform, so every stream has a fixed size. The network has one small
"tower" per stream, a fusion block that mixes the towers (either a learned
gate, "gate", or a plain average after attention, "core"), an optional
global skip connection, and an ordinal CORAL output head.

The pipeline (towers, fusion, CORAL head, losses, hyper-parameters and
stream sizes) is copied unchanged from the DIPSER gTAO-Net grid.

DATA IT EXPECTS (Kaggle inputs)
-------------------------------
Kaggle datasets "laavanayadhawan/engagenet-personalization" and
"laavanayadhawan/openface-engagenet", mounted under /kaggle/input. Face crops
live at paths like

    .../{Train,Validation,Test}/<class name>/subject_<id>_..._vid_<a>_<b>_frame_<f>.jpg

The <class name> folder gives the label. The features are extracted once and
cached to "engagenet_clips_v1.npz".

THE SCOPED CONFIGURATION GRID (6 configs, protocol P2)
------------------------------------------------------
    unimodal_face, unimodal_facemesh, unimodal_head_pose  (one stream each)
    gate_skip1_drop0                                      (gated fusion)
    core_skip1_drop0                                      (attention-average fusion)
    shuffle_core_skip1_drop1                              (permuted-label control)

The permuted-label control trains on randomly shuffled labels. Its QWK must
stay near zero (|QWK| <= SHUFFLE_TOL = 0.10); otherwise something leaks.

PROTOCOLS
---------
    P2 (used): 5 folds grouped BY SUBJECT (no person is in both train and
        test) x 3 seeds {0, 1, 2}. Out-of-fold (OOF) predictions are pooled
        and scored with QWK (quadratic weighted kappa), accuracy and MAE.
    P1 (code kept, NOT run): the official Kaggle Train/Validation split. It
        is not subject-disjoint (every validation subject also appears in
        Train), so it was dropped. masked_eval() (stream masking, "H4") is
        also kept but not run.

NESTED EARLY STOPPING (what "v2" adds)
--------------------------------------
Early stopping keeps the model from the epoch with the best validation
score. If that validation set is the same fold that is scored at the end,
the reported score is too optimistic (this is the v1 flaw). v2 fixes it:
inside every training fold, INNER_FRAC = 20% of the TRAINING subjects are
held out as an inner validation set; the epoch is chosen on them, and the
outer fold is used only for the final, scored prediction. Its labels never
influence training or epoch selection. The subjects are re-drawn (up to 20
times) until inner-train AND inner-val both contain at least 2 classes.
Each fold prints a line
"    inner split: train n=..., val n=..., val subjects=..." in the log, and
its JSON fold record gets "n_inner_train", "n_inner_val" and
"inner_val_subjects" (nested mode only). Note that "best_val_qwk" /
"best_epoch" then refer to the INNER validation set.
GTAO_NESTED=1 (default) switches this on; GTAO_NESTED=0 restores v1 exactly.

Fix note: the automatic v1 -> v2 patch (tools/make_nested.py) first put the
nested guard into official_split_folds() instead of fit_fold() (fit_fold
had no docstring to anchor on), so nesting was silently inactive. The guard
now sits at the start of fit_fold, as in the DIPSER v2 script.

HOW TO RUN
----------
On Kaggle: push it as a "script" kernel (GPU on, internet on, the two
datasets attached). It pip-installs mediapipe/opencv itself.

Locally (synthetic smoke test, no dataset needed):

    GRID_FAST=1 python engagenet-gtao-scoped.py           # 8 epochs, 400 clips max
    GRID_FAST=1 GTAO_NESTED=0 python engagenet-gtao-scoped.py

Without ENGAGENET_ROOT pointing at a folder of crops, a synthetic corpus of
200 random clips is generated instead (synth_corpus).

FILES IT WRITES (in /kaggle/working on Kaggle, /tmp/engagenet_gt_out locally)
----------------------------------------------------------------------------
    engagenet_clips_v1.npz                     per-clip stream feature cache
    engagenet_gt_metrics.json                  status, env, summary rows per protocol
    engagenet_gt_grid/P2_groupkfold/<config>/seed<k>/oof.json
                                               per-fold logs + pooled OOF predictions
"""

# %% [markdown]
# # gTAO-Net on EngageNet — 3-stream benchmark, scoped P2 (v5)
#
# Version history kept from the original notebook:
#   * P1 official split dropped (the Kaggle split is NOT subject-disjoint:
#     85 of 85 validation subjects recur in Train).
#   * v3: missing facemesh/head_pose frames are zero-filled, not NaN.
#   * v4: np.load returns a read-only NpzFile; load_corpus copies it to a dict.
#   * Compute-budgeted P2 subset (5 h session cap): 6 configs x 3 seeds.
# EngageNet gives one hard label per clip, so the `soft` loss equals `coral`
# here; it is kept only for grid parity.

# %%
# =============================================================================
# SECTION 1. Imports
# =============================================================================
import collections
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
except ImportError:  # local smoke test without opencv
    cv2 = None

# %%
# =============================================================================
# SECTION 2. Kaggle setup (pip install + protobuf shim), then late imports
# =============================================================================
ON_KAGGLE = os.path.isdir("/kaggle/working")
if ON_KAGGLE:
    os.system("pip -q install mpmath scikit-learn opencv-python-headless 'protobuf==4.25.3' 'mediapipe==0.10.14'")
    # Kaggle's protobuf 4.25.3 does not ship google.protobuf.runtime_version
    # (it only exists in protobuf>=5.28), but mediapipe's generated code
    # imports it. We write a tiny stand-in module so mediapipe keeps working
    # on the 4.25.x version it was built against.
    import pathlib as pathlib_module
    import google.protobuf as google_protobuf
    protobuf_package_dir = pathlib_module.Path(google_protobuf.__file__).parent
    runtime_version_file = protobuf_package_dir / "runtime_version.py"
    if not runtime_version_file.exists():
        runtime_version_file.write_text(
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
    print("PB_FILE", protobuf_package_dir / "__init__.py")
    print("PB_VER", google_protobuf.__version__)
    print("RV_EXISTS_AFTER_SHIM", runtime_version_file.exists())
    try:
        import mediapipe as mediapipe_module
        print("MP_OK_TOP", mediapipe_module.__version__)
    except Exception as mediapipe_error:
        print("MP_FAIL_TOP", repr(mediapipe_error)[:400])
        raise SystemExit(1) from mediapipe_error

# These packages may have just been installed above, so import them now.
import mpmath
import socket
import sklearn  # noqa: F401  (imported for parity with the original script)
from sklearn.metrics import cohen_kappa_score

# %%
# =============================================================================
# SECTION 3. Output paths and settings (hyper-parameters)
# =============================================================================
if ON_KAGGLE:
    OUT = Path("/kaggle/working")
else:
    OUT = Path("/tmp/engagenet_gt_out")
OUT.mkdir(parents=True, exist_ok=True)
METRICS = OUT / "engagenet_gt_metrics.json"     # summary of every protocol
CACHE = OUT / "engagenet_clips_v1.npz"          # per-clip stream tensors (reused)
RUNS = OUT / "engagenet_gt_grid"                # one oof.json per config/seed

FAST = os.environ.get("GRID_FAST", "") == "1"   # quick smoke-test mode
SEQ_K = 6                                       # 1-D Tchebichef order across time
FACE_SIZE = 64                                  # face crop resized to 64x64
FACE_K = 16                                     # 2-D Tchebichef order -> 16x16
N_CLASSES = 4
CLASS_ORDER = {"not-engaged": 0, "barely-engaged": 1, "engaged": 2, "highly-engaged": 3}
EMB = 64                                        # size of each tower's output
HID = 128                                       # hidden size inside a tower
LR = 3e-3                                       # Adam learning rate
if FAST:
    EPOCHS = 8
else:
    EPOCHS = 40
BATCH = 32
PATIENCE = 8                                    # early stopping patience (epochs)
# v2: nested early stopping. Epoch selection uses an inner validation split of
# the TRAINING-fold subjects; the scored outer fold never influences training.
NESTED = os.environ.get("GTAO_NESTED", "1") == "1"
INNER_FRAC = 0.2
MIN_EPOCHS = 5                                  # never stop before this epoch
N_FOLDS = 5
SEEDS = (0, 1, 2)
MDROP_P = 0.2                                   # modality-dropout probability

STREAMS = ("face", "facemesh", "head_pose")
FUSIONS = ("gate", "core")
SKIPS = (True, False)
MDROPS = (True, False)
LOSSES = ("coral", "hybrid", "soft")

SHUFFLE_TOL = 0.10                              # max |QWK| allowed for the control
if FAST:
    MAX_CLIPS = 400
else:
    MAX_CLIPS = None

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
print("device:", device, "| FAST:", FAST)

# %% [markdown]
# ## Basis + transforms (identical to gtao_grid / pipeline/tchebichef.py)

# %%
# =============================================================================
# SECTION 4. Tchebichef moments (the feature transform)
# =============================================================================
# Discrete Tchebichef polynomials t_n(x), x = 0..N-1, are orthogonal on a grid
# of N points. Projecting a signal on the first K of them gives K "moments"
# that summarise its shape (like a smooth, exact alternative to a DCT).
# They are built with the three-term recurrence
#     t_0(x) = 1
#     t_1(x) = 2x - N + 1
#     n t_n(x) = (2n-1)(2x-N+1) t_{n-1}(x) - (n-1)(N^2-(n-1)^2) t_{n-2}(x)
# and normalised by sqrt(rho(n, N)). High precision (mpmath, 50 digits) is
# used because the recurrence loses precision quickly in float64.

def _rho(n, N):
    """Squared norm of the order-n Tchebichef polynomial on N points.

    rho(n, N) = N * prod_{k=1..n} (N^2 - k^2) / (2n + 1)   (an mpmath number).
    """
    norm_value = mpmath.mpf(N)
    for k in range(1, n + 1):
        norm_value *= N * N - k * k
    return norm_value / (2 * n + 1)


def tchebichef_basis(N, K):
    """Return the orthonormal Tchebichef basis as a float32 array of shape (N, K).

    Column n holds the normalised polynomial t_n evaluated at x = 0..N-1.
    """
    with mpmath.workdps(50):
        # polynomials[n][x] = t_n(x), filled row by row
        polynomials = []
        for _ in range(K):
            polynomials.append([mpmath.mpf(0)] * N)
        for x in range(N):
            polynomials[0][x] = mpmath.mpf(1)
        if K >= 2:
            for x in range(N):
                polynomials[1][x] = mpmath.mpf(2 * x - N + 1)
        for n in range(2, K):
            a = mpmath.mpf(2 * n - 1)
            c = mpmath.mpf((n - 1) * (N * N - (n - 1) * (n - 1)))
            for x in range(N):
                polynomials[n][x] = (a * (2 * x - N + 1) * polynomials[n - 1][x]
                                     - c * polynomials[n - 2][x]) / n
        basis_matrix = np.zeros((N, K), dtype=np.float32)
        for n in range(K):
            norm = mpmath.sqrt(_rho(n, N))
            for x in range(N):
                basis_matrix[x, n] = float(polynomials[n][x] / norm)
    return basis_matrix


_BASIS = {}   # cache: (N, K) -> basis matrix, so each basis is built only once


def basis(N, K):
    """Return the (N, K) Tchebichef basis, computing it only on first use."""
    key = (N, K)
    if key not in _BASIS:
        _BASIS[key] = tchebichef_basis(N, K)
    return _BASIS[key]


def transform_2d(img, K):
    """2-D Tchebichef moments of a square image.

    img: (N, N) array. Returns (K, K) moments  M = B^T @ img @ B, B = basis(N, K).
    """
    basis_matrix = basis(img.shape[-1], K)
    return (basis_matrix.T @ img) @ basis_matrix


def transform_1d(x, K):
    """1-D Tchebichef moments along the last axis.

    x: (..., N) array. Returns (..., K) = x @ B.
    """
    basis_matrix = basis(x.shape[-1], K)
    return x @ basis_matrix


def pad_time(arr, k):
    """Make sure the time axis (axis 0) has at least k frames.

    Short sequences are padded by repeating the last frame. Raises
    ValueError if there are no frames at all.
    """
    if arr.shape[0] >= k:
        return arr
    if arr.shape[0] == 0:
        raise ValueError("empty time axis")
    repeated_last_frame = np.repeat(arr[-1:], k - arr.shape[0], axis=0)
    return np.concatenate([arr, repeated_last_frame], axis=0)

# %% [markdown]
# ## EngageNet loader: crops -> per-clip stream tensors (cached npz)
#
# The 4.15 GB dataset exposes face crops under
# `images_engagednet/ImageData3/{Train,Validation,Test}/<class>/subject_<id>_..._frame_<f>.jpg`.
# Face stream = Tchebichef moments of the crop (same front-end as DIPSER).
# Face-mesh = MediaPipe 478-refined-landmark mesh; head-pose = solvePnP
# (pitch, yaw, roll). Extraction runs once and caches to `CACHE`.

# %%
# =============================================================================
# SECTION 5. EngageNet data loading (real crops, cache, synthetic fallback)
# =============================================================================
# File name stem pattern, e.g. "subject_12_xyz_vid_3_4_frame_7".
# Groups: 1 = subject id, 2 = free text, 3/4 = video ids, 5 = frame number.
SUBJECT_RE = re.compile(
    r"^subject_(\d+)_(.*)_vid_(\d+)_(\d+)_frame_(\d+)$")


def crop_sort_key(crop_record):
    """Sort key for a crop record: its jpg path written with forward slashes."""
    jpg_path = crop_record[0]
    return jpg_path.as_posix()


def find_crops():
    """Find every labelled face crop on disk.

    Returns a list of (jpg_path, split, subject, clip) tuples sorted by path,
    where split is "Train", "Validation" or "Test", subject is
    "subject_<id>" and clip is the file name without ".jpg". Off Kaggle the
    folder in env var ENGAGENET_ROOT is searched.
    Quirk (kept from the original): if ENGAGENET_ROOT is NOT set, Path("")
    means the current directory, so every *.jpg below the current working
    directory is searched. Run local smoke tests from a folder without
    EngageNet-like jpgs to get the synthetic corpus.
    """
    if not ON_KAGGLE:
        root = Path(os.environ.get("ENGAGENET_ROOT", ""))
        if root.is_dir():
            search_dirs = [root]
        else:
            return []
    else:
        search_dirs = list(Path("/kaggle/input").glob("*"))
    found = []
    for base_dir in search_dirs:
        for jpg in base_dir.rglob("*.jpg"):
            path_text = str(jpg)
            split = None
            for split_name in ("Train", "Validation", "Test"):
                if f"/{split_name}/" in path_text or f"\\{split_name}\\" in path_text:
                    split = split_name
                    break
            match = SUBJECT_RE.match(jpg.stem)
            if split is None or match is None:
                continue
            subject = f"subject_{match.group(1)}"
            found.append((jpg, split, subject, jpg.stem))
    return sorted(found, key=crop_sort_key)


class MeshExtractor:
    """Wraps MediaPipe FaceMesh to get 478 landmarks and the head pose of a crop.

    If mediapipe cannot be loaded, self.fm stays None and detect() returns
    (None, None) for every image.
    """

    def __init__(self):
        """Create the MediaPipe FaceMesh detector (static images, one face)."""
        self.fm = None
        try:
            import mediapipe as mp
            self.mp = mp
            self.fm = mp.solutions.face_mesh.FaceMesh(
                static_image_mode=True, max_num_faces=1, refine_landmarks=True,
                min_detection_confidence=0.5)
        except Exception as error:  # pragma: no cover
            print("mediapipe unavailable:", repr(error))
            self.fm = None

    def detect(self, jpg):
        """Detect the face mesh in one jpg.

        Returns (mesh, pose): mesh is a float32 (478, 3) array of (x pixels,
        y pixels, relative depth z) and pose a float32 (3,) array of
        (pitch, yaw, roll) in degrees. Either can be None. If the first try
        fails, the image is padded with a white border (15% of its size) and
        tried once more, because tight crops sometimes hide the face outline.
        """
        if self.fm is None:
            return None, None
        img = cv2.cvtColor(cv2.imread(str(jpg)), cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]
        mesh, pose = None, None
        for attempt in range(2):
            if attempt == 0:
                pad = 0
            else:
                pad = int(0.15 * max(h, w))
            if pad:
                padded_image = cv2.copyMakeBorder(img, pad, pad, pad, pad,
                                                  cv2.BORDER_CONSTANT,
                                                  value=(255, 255, 255))
            else:
                padded_image = img
            result = self.fm.process(padded_image)
            if result.multi_face_landmarks:
                face_landmarks = result.multi_face_landmarks[0]
                # MediaPipe gives x, y in [0, 1]; convert them to pixels.
                point_list = []
                for landmark in face_landmarks.landmark:
                    point_list.append((landmark.x * padded_image.shape[1],
                                       landmark.y * padded_image.shape[0],
                                       landmark.z))
                points = np.array(point_list, dtype=np.float32)
                if pad:
                    points[:, :2] -= pad     # back to un-padded coordinates
                mesh = points
                pose = self._head_pose(points.astype(np.float64), w, h)
                break
        return mesh, pose

    @staticmethod
    def _head_pose(pts, w, h):
        """Estimate (pitch, yaw, roll) in degrees from 6 mesh landmarks.

        Uses cv2.solvePnP with a generic 3-D face model (nose tip, chin,
        eye corners, mouth corners) and a pinhole camera of focal length
        max(w, h). Returns a float32 (3,) array, or None on failure.
        """
        model_points_3d = np.array([
            [0.0, 0.0, 0.0], [0.0, -63.6, -12.5],
            [-43.3, 32.7, -26.0], [43.3, 32.7, -26.0],
            [-28.9, -28.9, -24.1], [28.9, -28.9, -24.1]], dtype=np.float64)
        landmark_indices = [1, 152, 33, 263, 61, 291]
        try:
            image_point_list = []
            for x, y, _ in pts[landmark_indices]:
                image_point_list.append([x, y])
            image_points_2d = np.array(image_point_list, dtype=np.float64)
            focal = max(w, h)
            camera_matrix = np.array([[focal, 0, w / 2], [0, focal, h / 2], [0, 0, 1]],
                                     dtype=np.float64)
            ok, rotation_vector, _ = cv2.solvePnP(model_points_3d, image_points_2d,
                                                  camera_matrix, None,
                                                  flags=cv2.SOLVEPNP_ITERATIVE)
            if not ok:
                return None
            R, _ = cv2.Rodrigues(rotation_vector)   # 3x3 rotation matrix
            sy = math.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
            pitch = math.atan2(-R[2, 1], R[2, 2])
            yaw = math.atan2(R[2, 0], sy)
            roll = math.atan2(-R[1, 0], R[0, 0])
            return np.array([math.degrees(pitch), math.degrees(yaw),
                             math.degrees(roll)], dtype=np.float32)
        except Exception:
            return None


def encode_clip(jpg, split, subj, clip, label, ext):
    """Encode ONE frame (one jpg) of a clip.

    Returns a dict with the face moments (256 = 16x16 values), the mesh
    (478, 3) and head pose (3,) from `ext` (None if unavailable), and the
    integer label. Despite its name it works per frame; load_index groups
    the frames of a clip afterwards.
    """
    grey = Image.open(jpg).convert("L").resize((FACE_SIZE, FACE_SIZE))
    img = np.asarray(grey, dtype=np.float32) / 255.0                      # (64, 64)
    face = transform_2d(img, FACE_K).ravel().astype(np.float32)          # 256
    mesh, pose = None, None
    if ext is not None:
        mesh, pose = ext.detect(jpg)
    return {"key": f"{split}/{subj}/{clip}", "subject": subj, "split": split,
            "face": face, "facemesh": mesh, "head_pose": pose,
            "label": CLASS_ORDER[label]}


def frame_number_of_entry(entry):
    """Sort key: the frame number at the end of an entry's key ("..._frame_<f>")."""
    return int(re.search(r"_frame_(\d+)$", entry["key"]).group(1))


def stack_present_frames(entries, stream_key):
    """Stack the non-None per-frame arrays of one stream into (T, ...) array.

    Raises (like np.stack) if no frame has this stream.
    """
    present_frames = []
    for entry in entries:
        frame_value = entry[stream_key]
        if frame_value is not None:
            present_frames.append(frame_value)
    return np.stack(present_frames)


def temporal_moments(frames):
    """Turn a (T, ...) per-frame array into one flat clip vector.

    Steps: pad to at least SEQ_K frames, move time to the last axis, flatten
    the other axes -> (D, T), then 1-D Tchebichef over time -> (D, SEQ_K),
    flattened to D * SEQ_K numbers.
    Caveat (kept from the original): clips with MORE than SEQ_K frames are
    not cut to SEQ_K, so reshape(-1, SEQ_K) then mixes frames together and
    the vector gets longer than D * SEQ_K.
    """
    padded = pad_time(frames, SEQ_K)
    time_last = np.moveaxis(padded, 0, -1).reshape(-1, SEQ_K)
    return transform_1d(time_last, SEQ_K).ravel()


def load_index():
    """Encode every available crop once; cache clip-level tensors to npz.

    Clip stream vector = pad_time to SEQ_K then a 1-D temporal Tchebichef, i.e.
    identical stream dims to DIPSER: face 1536, facemesh 8604, head_pose 18.
    Returns the opened npz file. If no crops are found, a synthetic corpus
    is generated instead (local smoke test).
    """
    if CACHE.exists():
        print("loading cached clip corpus:", CACHE)
        return np.load(CACHE, allow_pickle=True)

    crops = find_crops()
    if not crops:
        print("no EngageNet crops found — synthetic smoke fallback")
        return synth_corpus()

    # The label is the name of the folder that contains the jpg.
    label_of = {}
    for jpg, split, subj, clip in crops:
        try:
            label_of[(split, clip)] = jpg.parent.name.lower()
        except Exception:
            continue
    if not label_of:
        raise SystemExit("no labelled EngageNet crops")

    # per_frame[(split, clip)] = list of frame dicts from encode_clip
    per_frame = collections.defaultdict(list)
    ext = None
    if cv2 is not None:
        ext = MeshExtractor()
        if ext.fm is None:
            ext = None
    t0 = time.time()
    for n, (jpg, split, subj, clip) in enumerate(crops):
        if (split, clip) not in label_of:
            continue
        if MAX_CLIPS and len(per_frame) >= MAX_CLIPS:
            break
        entry = encode_clip(jpg, split, subj, clip, label_of[(split, clip)], ext)
        per_frame[(split, clip)].append(entry)
        if (n + 1) % 2000 == 0:
            print(f"  {n + 1}/{len(crops)} crops, {len(per_frame)} clips "
                  f"[{time.time() - t0:.0f}s]", flush=True)

    # One record per clip: frames sorted by frame number, then temporal moments.
    records = []
    for (split, clip), entries in per_frame.items():
        entries.sort(key=frame_number_of_entry)
        first = entries[0]
        label = first["label"]
        face_list = []
        for entry in entries:
            face_list.append(entry["face"])
        faces = np.stack(face_list)                                        # (T, 256)
        clip_features = {"face": temporal_moments(faces)}
        for stream_key in ("facemesh", "head_pose"):
            arr = None
            try:
                arr = stack_present_frames(entries, stream_key)
            except Exception:
                arr = None
            if arr is not None and arr.shape[0] > 0:
                clip_features[stream_key] = temporal_moments(arr)
        soft = np.zeros(N_CLASSES, dtype=np.float32)   # one-hot "soft" label
        soft[label] = 1.0
        record = {"key": first["key"], "subject": first["subject"],
                  "split": first["split"], "label": label, "soft": soft}
        for feature_name, feature_vector in clip_features.items():
            record[feature_name] = feature_vector
        records.append(record)

    key_list = []
    subject_list = []
    split_list = []
    label_list = []
    soft_list = []
    for record in records:
        key_list.append(record["key"])
        subject_list.append(record["subject"])
        split_list.append(record["split"])
        label_list.append(record["label"])
        soft_list.append(record["soft"])
    keys = np.asarray(key_list)
    subjects = np.asarray(subject_list)
    splits = np.asarray(split_list)
    y = np.asarray(label_list, dtype=np.int64)
    soft = np.stack(soft_list).astype(np.float32)

    # avail[s][i] = True if clip i has stream s
    avail = {}
    for stream_name in STREAMS:
        present_flags = []
        for record in records:
            present_flags.append(record.get(stream_name) is not None)
        avail[stream_name] = np.asarray(present_flags, dtype=bool)

    # feats[s] = (n_clips, dim) float32 matrix; clips without s stay all-zero.
    feats = {}
    for stream_name in STREAMS:
        feats[stream_name] = None
    for stream_name in STREAMS:
        present_indices = []
        for i, record in enumerate(records):
            if record.get(stream_name) is not None:
                present_indices.append(i)
        if present_indices:
            present_vectors = []
            for i in present_indices:
                present_vectors.append(records[i][stream_name])
            arr = np.asarray(present_vectors, dtype=np.float32)
            full = np.full((len(records), arr.shape[1]), 0.0, dtype=np.float32)
            full[present_indices] = arr
            feats[stream_name] = full
        if len(records) == 0:
            denominator = 1
        else:
            denominator = len(records)
        print(f"stream {stream_name}: present in {len(present_indices)}/{len(records)} "
              f"({100 * len(present_indices) / denominator:.1f}%)")

    # A stream that is missing everywhere is stored as an empty (0, 0) array.
    empty2d = np.zeros((0, 0), dtype=np.float32)
    saved_features = {}
    for stream_name in ("face", "facemesh", "head_pose"):
        if feats[stream_name] is not None:
            saved_features[stream_name] = feats[stream_name]
        else:
            saved_features[stream_name] = empty2d
    np.savez(CACHE, keys=keys, subjects=subjects, splits=splits, y=y, soft=soft,
             avail_face=avail["face"], avail_facemesh=avail["facemesh"],
             avail_head_pose=avail["head_pose"],
             feat_face=saved_features["face"],
             feat_facemesh=saved_features["facemesh"],
             feat_head_pose=saved_features["head_pose"])
    print(f"corpus: {len(records)} clips, {len(set(subjects.tolist()))} subjects, "
          f"{time.time() - t0:.0f}s")
    return np.load(CACHE, allow_pickle=True)


def load_corpus():
    """load_index + sanitize: missing frames are zero-filled, never NaN/Inf.

    Bug fix (v3): v1 wrote np.nan into the slots of absent facemesh/head_pose
    crops (1.6% of clips), poisoning every model that read those streams —
    all-NaN logits collapsed to the constant class-0 predictor. Zero-fill the
    missing slots and defensively nan_to_num any residual non-finite value.

    v5: np.load returns a read-only NpzFile; materialize a plain dict first so
    the sanitized features can be written back (v4 crashed on item assignment).

    Returns a dict: keys, subjects, splits, y, soft, avail_<stream>, feat_<stream>.
    """
    corpus = load_index()
    out = {}
    for array_name in corpus.files:
        out[array_name] = corpus[array_name]
    for stream_name in STREAMS:
        features = out[f"feat_{stream_name}"]
        if features.shape[0] > 0:
            out[f"feat_{stream_name}"] = np.nan_to_num(features, nan=0.0, posinf=0.0,
                                                       neginf=0.0)
    return out


def synth_corpus():
    """Local GRID_FAST smoke: class-dependent per-subject random faces.

    200 fake clips, 20 subjects (subjects 0-3 are "Validation"), labels
    cycling 0,1,2,3. Each stream is Gaussian noise shifted by the label, so
    a model can learn something. Written to CACHE, then re-opened.
    """
    rng = np.random.default_rng(0)
    n = 200
    records = []
    val_subjects = set()
    for i in range(4):
        val_subjects.add(f"subject_{i}")
    for i in range(n):
        subj = f"subject_{i % 20}"
        if subj in val_subjects:
            split = "Validation"
        else:
            split = "Train"
        label = int(i % N_CLASSES)
        # The random draws must stay in this order: face, mesh, head pose.
        face = rng.normal(size=(1536,)).astype(np.float32)
        face += 0.1 * label
        mesh = rng.normal(size=(8604,)).astype(np.float32) + 0.05 * label
        hp = rng.normal(size=(18,)).astype(np.float32) + 0.2 * label
        soft = np.zeros(N_CLASSES, dtype=np.float32)
        soft[label] = 1.0
        records.append({"key": f"{split}/{subj}/{i}", "subject": subj,
                        "split": split, "label": label, "soft": soft,
                        "face": face, "facemesh": mesh, "head_pose": hp})
    key_list = []
    subject_list = []
    split_list = []
    label_list = []
    soft_list = []
    face_list = []
    mesh_list = []
    head_pose_list = []
    for record in records:
        key_list.append(record["key"])
        subject_list.append(record["subject"])
        split_list.append(record["split"])
        label_list.append(record["label"])
        soft_list.append(record["soft"])
        face_list.append(record["face"])
        mesh_list.append(record["facemesh"])
        head_pose_list.append(record["head_pose"])
    keys = np.asarray(key_list)
    subjects = np.asarray(subject_list)
    splits = np.asarray(split_list)
    y = np.asarray(label_list, dtype=np.int64)
    soft = np.stack(soft_list).astype(np.float32)
    np.savez(CACHE, keys=keys, subjects=subjects, splits=splits, y=y, soft=soft,
             avail_face=np.ones(n, dtype=bool), avail_facemesh=np.ones(n, dtype=bool),
             avail_head_pose=np.ones(n, dtype=bool),
             feat_face=np.stack(face_list),
             feat_facemesh=np.stack(mesh_list),
             feat_head_pose=np.stack(head_pose_list))
    return np.load(CACHE, allow_pickle=True)

# %% [markdown]
# ## Grid model (copy of gtao_grid.py: gate / CORE fusion, skip, mdrop, CORAL)

# %%
# =============================================================================
# SECTION 6. The gTAO-Net model
# =============================================================================
# Shapes used below: B = batch size, S = number of streams, E = EMB = 64.

class StreamTower(nn.Module):
    """Small MLP that turns one stream vector into an EMB-sized embedding.

    (B, in_dim) -> Linear -> GELU -> Linear -> LayerNorm -> (B, EMB)
    """

    def __init__(self, in_dim, hidden=HID, emb=EMB):
        """Build the two linear layers and the LayerNorm."""
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, hidden), nn.GELU(),
                                 nn.Linear(hidden, emb), nn.LayerNorm(emb))

    def forward(self, x):
        """x: (B, in_dim) -> (B, EMB)."""
        return self.net(x)


class GatedCrossModalFusion(nn.Module):
    """"gate" fusion: cross-stream attention, then a learned weight per stream.

    1. Attention: every stream looks at every other stream,
       attn = softmax(Q K^T / sqrt(E)), z2 = attn @ z           (B, S, E)
    2. Gating: a small MLP reads all of z2 and outputs S weights
       alpha (softmax, they sum to 1). fused = sum_s alpha_s * z2_s  (B, E)
    alpha is returned so we can see which stream the model relies on.
    """

    def __init__(self, n_streams, emb=EMB):
        """Create query/key projections and the gate MLP."""
        super().__init__()
        self.proj_q = nn.Linear(emb, emb, bias=False)
        self.proj_k = nn.Linear(emb, emb, bias=False)
        self.gate_net = nn.Sequential(nn.Linear(n_streams * emb, n_streams * emb),
                                      nn.GELU(), nn.Linear(n_streams * emb, n_streams))

    def forward(self, z):
        """z: (B, S, E) -> (fused (B, E), alpha (B, S))."""
        q = self.proj_q(z)
        k = self.proj_k(z)
        scores = q @ k.transpose(1, 2) / math.sqrt(z.shape[-1])    # (B, S, S)
        attn = torch.softmax(scores, dim=-1)
        z2 = attn @ z                                              # (B, S, E)
        gate_input = z2.reshape(z2.shape[0], -1)                   # (B, S*E)
        alpha = torch.softmax(self.gate_net(gate_input), dim=-1)   # (B, S)
        fused = (alpha.unsqueeze(-1) * z2).sum(dim=1)              # (B, E)
        return fused, alpha


class CoreAttentionFusion(nn.Module):
    """"core" fusion: the same cross-stream attention, then a plain mean.

    No gate, so alpha is None.
    """

    def __init__(self, n_streams, emb=EMB):
        """Create query/key projections (n_streams is unused, kept for symmetry)."""
        super().__init__()
        self.proj_q = nn.Linear(emb, emb, bias=False)
        self.proj_k = nn.Linear(emb, emb, bias=False)

    def forward(self, z):
        """z: (B, S, E) -> (mean over streams (B, E), None)."""
        q = self.proj_q(z)
        k = self.proj_k(z)
        scores = q @ k.transpose(1, 2) / math.sqrt(z.shape[-1])    # (B, S, S)
        attn = torch.softmax(scores, dim=-1)
        z2 = attn @ z                                              # (B, S, E)
        return z2.mean(dim=1), None


class GlobalSkip(nn.Module):
    """Global skip connection: a linear map of each RAW stream, summed.

    skip = sum_s W_s x_s, shape (B, EMB). It lets raw features bypass the
    towers and the fusion.
    """

    def __init__(self, stream_dims, emb=EMB):
        """One bias-free Linear(dim_s, EMB) per stream, in stream_dims order."""
        super().__init__()
        projections = {}
        for stream_name, stream_dim in stream_dims.items():
            projections[stream_name] = nn.Linear(stream_dim, emb, bias=False)
        self.proj = nn.ModuleDict(projections)

    def forward(self, streams):
        """streams: dict name -> (B, dim). Returns (B, EMB)."""
        projected = []
        for stream_name, stream_tensor in streams.items():
            projected.append(self.proj[stream_name](stream_tensor))
        return torch.stack(projected).sum(dim=0)


class CORALHead(nn.Module):
    """CORAL ordinal output head for N_CLASSES ordered classes.

    Instead of 4 independent class scores it predicts 3 binary questions
    "is y > 0?", "is y > 1?", "is y > 2?". All three share one weight
    vector w and differ only by their bias b_k:  logit_k = w.x + b_k.
    Sharing w keeps the answers consistent (rank-consistent).
    """

    def __init__(self, emb=EMB, num_classes=N_CLASSES):
        """Shared Linear(EMB, 1) plus num_classes-1 biases (0.5 ... -0.5)."""
        super().__init__()
        self.d = num_classes - 1
        self.w = nn.Linear(emb, 1)
        self.b = nn.Parameter(torch.linspace(0.5, -0.5, self.d))

    def forward(self, x):
        """x: (B, EMB) -> logits (B, d)."""
        return self.w(x) + self.b

    def probs(self, logits):
        """Turn the d logits into class probabilities (B, num_classes).

        With p_k = P(y > k) = sigmoid(logit_k):
            P(y = c) = P(y > c-1) - P(y > c),  using P(y > -1) = 1, P(y > d) = 0.
        Negative values are clipped to 0 and rows renormalised to sum to 1.
        """
        prob_greater = torch.sigmoid(logits)
        extended = torch.cat([torch.ones_like(prob_greater[:, :1]), prob_greater,
                              torch.zeros_like(prob_greater[:, :1])], dim=1)
        p = (extended[:, :-1] - extended[:, 1:]).clamp(min=0)
        return p / p.sum(dim=1, keepdim=True)

    def loss(self, logits, y):
        """Binary cross-entropy on the d questions; target_k = 1 if y > k."""
        k = torch.arange(self.d, device=logits.device)[None, :]
        targets = (y[:, None] > k).float()
        return F.binary_cross_entropy_with_logits(logits, targets)

    def predict(self, logits):
        """Most likely class index per row (B,), from the same maths as probs()."""
        prob_greater = torch.sigmoid(logits)
        extended = torch.cat([torch.ones_like(prob_greater[:, :1]), prob_greater,
                              torch.zeros_like(prob_greater[:, :1])], dim=1)
        p = (extended[:, :-1] - extended[:, 1:]).clamp(min=0)
        return (p / p.sum(dim=1, keepdim=True)).argmax(dim=1)


def head_loss(head, logits, y, soft, loss="coral"):
    """Training loss for one batch.

    loss = "coral":  plain CORAL binary cross-entropy.
    loss = "soft":   CORAL BCE against targets 1 - cumsum(soft label).
    otherwise ("hybrid"):  CE + 0.5 * CORAL BCE + 0.1 * smoothed CE, where the
                     smoothed target puts 0.7 on the true class and 0.15 on
                     each existing neighbour class (then renormalised).
    """
    if loss == "coral":
        return head.loss(logits, y)
    if loss == "soft":
        cumulative = torch.cumsum(soft, dim=1)[:, :head.d]
        return F.binary_cross_entropy_with_logits(logits, 1.0 - cumulative)
    p = head.probs(logits)
    probability_of_true_class = p.gather(1, y[:, None]).squeeze(1)
    ce = -torch.log(probability_of_true_class + 1e-8).mean()
    bce = head.loss(logits, y)
    smoothed_target = torch.zeros_like(p)
    smoothed_target.scatter_(1, y[:, None], 0.7)
    for offset in (-1, 1):
        neighbour = y + offset
        rows = torch.nonzero((neighbour >= 0) & (neighbour < N_CLASSES)).squeeze(1)
        if rows.numel():
            smoothed_target[rows, neighbour[rows].clamp(0, N_CLASSES - 1)] += 0.15
    smoothed_target = smoothed_target / smoothed_target.sum(dim=1, keepdim=True)
    smooth = -(smoothed_target * torch.log(p + 1e-8)).sum(dim=1).mean()
    return ce + 0.5 * bce + 0.1 * smooth


class GTAONetGrid(nn.Module):
    """The full gTAO-Net used in the grid.

    towers (one per stream) -> stack (B, S, E) -> optional modality dropout
    -> fusion ("gate" or "core") -> + optional global skip -> CORAL head.
    Sub-modules are created in this order: towers, fusion, skip, head
    (the order matters for random weight initialisation).
    """

    def __init__(self, stream_dims, fusion="gate", use_skip=True, mdrop=False,
                 emb=EMB):
        """stream_dims: dict stream name -> input size."""
        super().__init__()
        self.streams = sorted(stream_dims)          # stream names, alphabetical
        towers = {}
        for stream_name in self.streams:
            towers[stream_name] = StreamTower(stream_dims[stream_name])
        self.towers = nn.ModuleDict(towers)
        if fusion == "gate":
            self.fusion = GatedCrossModalFusion(len(self.streams))
        else:
            self.fusion = CoreAttentionFusion(len(self.streams))
        if use_skip:
            self.skip = GlobalSkip(stream_dims)
        else:
            self.skip = None
        self.head = CORALHead()
        self.use_skip = use_skip
        self.mdrop = mdrop
        self.p = MDROP_P

    def forward(self, streams, apply_mdrop=None):
        """streams: dict name -> (B, dim). Returns (logits (B, 3), alpha or None).

        Modality dropout: during training (if enabled) each stream embedding
        of each sample is zeroed with probability MDROP_P, so the model
        learns not to depend on a single stream.
        """
        embeddings = []
        for stream_name in self.streams:
            embeddings.append(self.towers[stream_name](streams[stream_name]))
        z = torch.stack(embeddings, dim=1)                      # (B, S, E)
        if (self.training and self.mdrop) or apply_mdrop:
            keep_mask = torch.rand(z.shape[0], z.shape[1], device=z.device) > self.p
            z = z * keep_mask.unsqueeze(-1).float()
        fused, alpha = self.fusion(z)                            # (B, E)
        if self.use_skip and self.skip is not None:
            fused = fused + self.skip(streams)
        return self.head(fused), alpha

# %% [markdown]
# ## Metrics + training helpers (gtao_grid.py)

# %%
# =============================================================================
# SECTION 7. Metrics and training helpers
# =============================================================================
def qwk(y_true, y_pred):
    """Quadratic weighted kappa (QWK) between true and predicted classes.

    QWK = 1 - sum(W * O) / sum(W * E), with O the confusion matrix, E the
    matrix expected by chance and W_ij = (i - j)^2: big mistakes (0 vs 3)
    cost more than small ones. 1 = perfect, 0 = chance level.
    Returns 0.0 when it is undefined (fewer than 2 samples or classes).
    """
    if len(y_true) < 2 or len(np.unique(y_true)) < 2:
        return 0.0
    try:
        value = float(cohen_kappa_score(y_true, y_pred, weights="quadratic"))
    except ValueError:
        value = 0.0
    if not math.isfinite(value):
        return 0.0
    return value


def to_torch(streams_dict):
    """Convert a dict of numpy arrays into a dict of tensors on `device`."""
    tensors = {}
    for stream_name, array in streams_dict.items():
        tensors[stream_name] = torch.as_tensor(array, device=device)
    return tensors


@torch.no_grad()
def evaluate(model, val_streams, val_labels):
    """Score the model (no dropout) on a validation set: {"qwk", "acc", "mae"}."""
    logits, _ = model(val_streams, apply_mdrop=False)
    predictions = model.head.predict(logits).cpu().numpy()
    labels = val_labels.cpu().numpy()
    return {"qwk": qwk(labels, predictions),
            "acc": round(float((predictions == labels).mean()), 4),
            "mae": round(float(np.abs(predictions - labels).mean()), 4)}


def set_seed(seed):
    """Seed torch and numpy's global random generators."""
    torch.manual_seed(seed)
    np.random.seed(seed)


def select_rows(stream_tensors, row_mask):
    """Return a new dict with only the rows of each tensor where row_mask is True."""
    selected = {}
    for stream_name, tensor in stream_tensors.items():
        selected[stream_name] = tensor[row_mask]
    return selected


def _draw_inner_val_mask(subj_tr, train_labels):
    """Choose which training rows form the inner validation set.

    Draws round(INNER_FRAC * n_subjects) (at least 1) training SUBJECTS at
    random with numpy's global RNG, re-drawing (at most 20 times) until BOTH
    sides (inner-train and inner-val) contain at least 2 classes. (An
    inner-val set with a single class always has QWK 0, so epoch selection
    would degenerate.) Raises RuntimeError if no usable draw is found.
    Returns (is_val, val_subjects): a numpy bool array, one entry per
    training row, and the set of chosen subject ids. Only training-fold
    subjects can be chosen, so the outer (scored) fold is never touched.
    """
    subject_of_row = np.asarray(subj_tr).astype(str)
    unique_subjects = np.array(sorted(set(subject_of_row.tolist())))
    n_val = max(1, int(round(INNER_FRAC * len(unique_subjects))))
    labels_np = train_labels.cpu().numpy()
    found_usable_split = False
    for _ in range(20):                      # re-draw until both sides are usable
        shuffled_order = np.random.permutation(len(unique_subjects))
        val_subjects = set(unique_subjects[shuffled_order[:n_val]].tolist())
        is_val_list = []
        for subject in subject_of_row:
            is_val_list.append(subject in val_subjects)
        is_val_np = np.array(is_val_list)
        n_classes_inner_train = len(np.unique(labels_np[~is_val_np]))
        n_classes_inner_val = len(np.unique(labels_np[is_val_np]))
        if n_classes_inner_train >= 2 and n_classes_inner_val >= 2:
            found_usable_split = True
            break
    if not found_usable_split:
        raise RuntimeError("could not draw a usable inner split")
    return is_val_np, val_subjects


def _inner_split(Xtr, ytr, softtr, subj_tr):
    """Split the training fold by subject: (1-INNER_FRAC) train, INNER_FRAC val.

    Inputs: training streams dict, labels tensor, soft labels (or None) and
    the subject id of every training row. Prints one log line with the
    sizes. Returns
    (inner_train_streams, inner_train_labels, inner_train_soft,
     inner_val_streams, inner_val_labels, split_summary)
    where split_summary = {"n_inner_train", "n_inner_val",
    "inner_val_subjects" (sorted list)} is stored in the fold's JSON record.
    """
    is_val_np, val_subjects = _draw_inner_val_mask(subj_tr, ytr)
    is_val = torch.as_tensor(is_val_np, device=ytr.device)
    if softtr is not None:
        inner_train_soft = softtr[~is_val]
    else:
        inner_train_soft = None
    split_summary = {"n_inner_train": int((~is_val_np).sum()),
                     "n_inner_val": int(is_val_np.sum()),
                     "inner_val_subjects": sorted(val_subjects)}
    print(f"    inner split: train n={split_summary['n_inner_train']}, "
          f"val n={split_summary['n_inner_val']}, val subjects={len(val_subjects)}",
          flush=True)
    return (select_rows(Xtr, ~is_val), ytr[~is_val], inner_train_soft,
            select_rows(Xtr, is_val), ytr[is_val], split_summary)


def fit_fold(model, train_streams, train_labels, val_streams, val_labels, stream_list,
             loss="coral", train_soft=None, subj_tr=None):
    """Train `model` on one fold with Adam and early stopping.

    Each epoch: shuffle, mini-batches of BATCH, clip gradients to norm 1.
    After each epoch the model is scored on (val_streams, val_labels); the
    weights of the best epoch (ties -> later epoch) are kept. Training stops
    once MIN_EPOCHS are done and PATIENCE epochs pass without improvement.
    Every 5 epochs the mean gate weights on the validation set are logged.

    Nested mode (NESTED, the default): the callers pass the OUTER fold as
    (val_streams, val_labels), but it is replaced right away by an inner
    validation split of the training subjects (subj_tr = subject id of
    every training row, required). So the outer fold is never used here;
    the caller scores it afterwards. With GTAO_NESTED=0 the outer fold is
    used for epoch selection, exactly as in v1.

    Returns (best, trace, info):
        best  = {"qwk", "state" (weights), "epoch"}
        trace = list of {"epoch", "val_qwk", "gate"}
        info  = {"epochs" (epochs run), "loss_last" (mean of last 64 batch losses)}
                plus, in nested mode only, "inner_split" = the split summary
                from _inner_split (n_inner_train, n_inner_val, inner_val_subjects).
    """
    split_summary = None
    if NESTED:
        if subj_tr is None:
            raise ValueError("NESTED=1 requires subj_tr (training-fold subject ids)")
        (train_streams, train_labels, train_soft,
         val_streams, val_labels, split_summary) = _inner_split(
            train_streams, train_labels, train_soft, subj_tr)
    set_seed(int(np.random.randint(2 ** 31)))           # fold-rooted seed
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    best = {"qwk": -1.0, "state": None, "epoch": 0}
    trace = []
    losses = []
    for epoch in range(1, EPOCHS + 1):
        model.train()
        order = torch.randperm(len(train_labels))
        for start in range(0, len(train_labels), BATCH):
            batch_index = order[start:start + BATCH]
            optimizer.zero_grad()
            batch_streams = {}
            for stream_name in stream_list:
                batch_streams[stream_name] = train_streams[stream_name][batch_index]
            logits, _ = model(batch_streams)
            batch_labels = train_labels[batch_index]
            if train_soft is not None:
                batch_soft = train_soft[batch_index]
            else:
                batch_soft = None
            loss_value = head_loss(model.head, logits, batch_labels, batch_soft, loss)
            loss_value.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(loss_value.item())
        model.eval()
        val_scores = evaluate(model, val_streams, val_labels)
        if val_scores["qwk"] > best["qwk"] - 1e-9:
            state_copy = {}
            for param_name, param_value in model.state_dict().items():
                state_copy[param_name] = param_value.detach().clone()
            best = {"qwk": val_scores["qwk"],
                    "state": state_copy,
                    "epoch": epoch}
        if epoch % 5 == 0:
            all_val_streams = {}
            for stream_name in stream_list:
                all_val_streams[stream_name] = val_streams[stream_name]
            with torch.no_grad():
                _, alpha = model(all_val_streams, apply_mdrop=False)
            gate = []
            if alpha is not None:
                for mean_weight in alpha.mean(dim=0).cpu().numpy():
                    gate.append(round(float(mean_weight), 4))
            trace.append({"epoch": epoch, "val_qwk": val_scores["qwk"], "gate": gate})
        if epoch >= MIN_EPOCHS and epoch - best["epoch"] >= PATIENCE:
            break
    model.load_state_dict(best["state"])
    model.eval()
    info = {"epochs": epoch, "loss_last": round(float(np.mean(losses[-64:])), 4)}
    if split_summary is not None:
        info["inner_split"] = split_summary
    return best, trace, info


def cfg_name(cfg):
    """Folder / report name of a config dict, e.g. "gate_skip1_drop0"."""
    if cfg.get("unimodal"):
        return f"unimodal_{cfg['unimodal']}"
    if cfg.get("shuffle"):
        return "shuffle_core_skip1_drop1"
    base = f"{cfg['fusion']}_skip{int(cfg['use_skip'])}_drop{int(cfg['mdrop'])}"
    if cfg.get("loss", "coral") == "coral":
        return base
    return base + f"_{cfg['loss']}"

# %% [markdown]
# ## Grid drivers
#
# `run_protocol(corpus, folds_by_seed, cfg, seed)` trains `cfg` across the given
# `(train, val)` folds for one seed and pools OOF predictions. `folds_by_seed`
# maps seed → list of `(idx_train, idx_val)` pairs (P2: GroupKFold-5; P1: one
# subject-filtered train/val pair).

# %%
# =============================================================================
# SECTION 8. Protocol drivers (train every fold, pool out-of-fold predictions)
# =============================================================================
def pooled_scores(labels, predictions):
    """Return (qwk, acc, mae), each rounded to 4 decimals, for int arrays."""
    pooled_qwk = round(qwk(labels, predictions), 4)
    pooled_acc = round(float(np.mean(predictions == labels)), 4)
    pooled_mae = round(float(np.mean(np.abs(predictions - labels))), 4)
    return pooled_qwk, pooled_acc, pooled_mae


def run_protocol(corpus, folds, cfg, seed):
    """Train config `cfg` on every (train_index, test_index) fold for one seed.

    Each test fold is predicted by the model trained on the other folds; the
    predictions are pooled ("out-of-fold", OOF) and scored together.
    For the shuffle control the labels are first permuted (rng seeded with
    `seed`); OOF scores are still computed against the TRUE labels.
    Returns the meta dict written to oof.json, or None if no stream exists.
    """
    ys = corpus["y"]
    soft_full = corpus["soft"]
    if cfg.get("unimodal"):
        stream_list = [cfg["unimodal"]]
    else:
        stream_list = []
        for stream_name in STREAMS:
            if corpus[f"feat_{stream_name}"].shape[0] > 0:
                stream_list.append(stream_name)
    dims = {}
    for stream_name in stream_list:
        if corpus[f"feat_{stream_name}"].shape[0] > 0:
            dims[stream_name] = int(corpus[f"feat_{stream_name}"].shape[1])
    if not dims:
        return None

    idx_all = np.arange(len(ys))
    y_run = ys.copy()
    soft_run = soft_full.copy()
    if cfg.get("shuffle"):
        rng = np.random.default_rng(seed)
        permutation = rng.permutation(len(idx_all))
        y_run[idx_all] = ys[idx_all][permutation]
        soft_run[idx_all] = soft_full[idx_all][permutation]
        print("shuffle control: labels permuted within whole population")

    oof_pred = np.full(len(ys), np.nan)     # NaN = clip not predicted (yet)
    meta = {"probe": None, "folds": [], "gate_trace": {}}
    for fold_number, (train_index, test_index) in enumerate(folds):
        # NOTE: `not test_index.any()` is also True when the only test index is 0
        # (kept as in the original).
        if len(set(y_run[train_index])) < 2 or not test_index.any():
            continue
        train_arrays = {}
        for stream_name in stream_list:
            train_arrays[stream_name] = corpus[f"feat_{stream_name}"][train_index]
        train_streams = to_torch(train_arrays)
        train_labels = torch.as_tensor(y_run[train_index], device=device)
        if cfg.get("loss") == "soft":
            train_soft = torch.as_tensor(soft_run[train_index], device=device)
        else:
            train_soft = None
        test_arrays = {}
        for stream_name in stream_list:
            test_arrays[stream_name] = corpus[f"feat_{stream_name}"][test_index]
        test_streams = to_torch(test_arrays)
        test_labels = torch.as_tensor(y_run[test_index], device=device)
        set_seed(int(f"{seed}{fold_number}"))   # e.g. seed 1, fold 3 -> 13
        if cfg.get("unimodal"):
            model = GTAONetGrid(dims, fusion="core", use_skip=False,
                                mdrop=False).to(device)
        else:
            model = GTAONetGrid(dims, fusion=cfg["fusion"], use_skip=cfg["use_skip"],
                                mdrop=cfg["mdrop"]).to(device)
        t0 = time.time()
        best, trace, fit_info = fit_fold(model, train_streams, train_labels,
                                         test_streams, test_labels,
                                         subj_tr=corpus["subjects"][train_index],
                                         stream_list=stream_list,
                                         loss=cfg.get("loss", "coral"),
                                         train_soft=train_soft)
        with torch.no_grad():
            logits, _ = model(test_streams, apply_mdrop=False)
            oof_pred[test_index] = model.head.predict(logits).cpu().numpy()
        meta["gate_trace"][fold_number] = trace
        fold_log = {"fold": fold_number, "n_train": int(len(train_index)),
                    "n_val": int(len(test_index)), "best_val_qwk": best["qwk"],
                    "best_epoch": best["epoch"]}
        for info_name, info_value in fit_info.items():
            if info_name != "inner_split":
                fold_log[info_name] = info_value
        fold_log["wall_s"] = round(time.time() - t0, 1)
        # Nested mode only: record the inner split after the existing keys.
        if "inner_split" in fit_info:
            for summary_name, summary_value in fit_info["inner_split"].items():
                fold_log[summary_name] = summary_value
        meta["folds"].append(fold_log)
        print(f"  fold {fold_number}: val n={len(test_index)} QWK={best['qwk']:.4f} "
              f"ep={best['epoch']}  [{round(time.time() - t0, 1)}s]", flush=True)

    predicted = np.isfinite(oof_pred)       # clips that received a prediction
    true_labels = ys[predicted].astype(int)
    predicted_labels = oof_pred[predicted].astype(int)
    meta["oof"] = {"keys": corpus["keys"][predicted].astype(str).tolist(),
                   "y": ys[predicted].astype(int).tolist(),
                   "pred": oof_pred[predicted].astype(int).tolist(),
                   "subject": corpus["subjects"][predicted].astype(str).tolist(),
                   "split": corpus["splits"][predicted].astype(str).tolist()}
    pooled_qwk, pooled_acc, pooled_mae = pooled_scores(true_labels, predicted_labels)
    meta["pooled"] = {"qwk": pooled_qwk,
                      "acc": pooled_acc,
                      "mae": pooled_mae,
                      "n": int(predicted.sum()),
                      "n_subjects": len(set(corpus["subjects"][predicted].tolist()))}
    return meta


def subject_group_folds(corpus, seed, n_folds=N_FOLDS):
    """P2 folds: shuffle the subjects (rng seeded with `seed`), deal them round-robin.

    Returns a list of n_folds (train_index, test_index) pairs; all clips of a
    subject land in the same fold, so train and test never share a person.
    """
    unique_subjects = sorted(set(corpus["subjects"].tolist()))
    rng = np.random.default_rng(seed)
    rng.shuffle(unique_subjects)
    fold_of_subject = {}
    for position, subject in enumerate(unique_subjects):
        fold_of_subject[subject] = position % n_folds
    fold_of_clip_list = []
    for subject in corpus["subjects"]:
        fold_of_clip_list.append(fold_of_subject[subject])
    fold_of_clip = np.array(fold_of_clip_list)
    folds = []
    for fold_number in range(n_folds):
        folds.append((np.flatnonzero(fold_of_clip != fold_number),
                      np.flatnonzero(fold_of_clip == fold_number)))
    return folds


def official_split_folds(corpus):
    """P1: train on Train minus subjects seen in Validation; eval on Validation.

    NOT USED (P1 is not run). Returns one (train_index, val_index) pair.
    """
    splits = corpus["splits"]
    train_mask = splits == "Train"
    val_mask = splits == "Validation"
    leak_subjects = set(corpus["subjects"][val_mask].tolist())
    not_leaked_list = []
    for subject in corpus["subjects"]:
        not_leaked_list.append(subject not in leak_subjects)
    clean_train_mask = train_mask & np.array(not_leaked_list)
    return [(np.flatnonzero(clean_train_mask), np.flatnonzero(val_mask))]


def summary_sort_key(row):
    """Sort summary rows by QWK (best first), then by config name."""
    return (-row["qwk_mean"], row["config"])


def row_qwk_mean(row):
    """Key function: a summary row's mean QWK."""
    return row["qwk_mean"]


def summarize(configs, doc_key, run_dir, write_to):
    """Read every seed's oof.json, build one summary row per config.

    Prints a table, checks the shuffle control against SHUFFLE_TOL, and
    stores the rows under write_to[doc_key]["rows"] (JSON). Returns the rows.
    """
    rows = []
    for cfg in configs:
        name = cfg_name(cfg)
        config_dir = run_dir / name
        qwks = []
        accs = []
        maes = []
        ns = []
        probes = []
        gates = []
        for seed in SEEDS:
            oof_file = config_dir / f"seed{seed}" / "oof.json"
            if not oof_file.exists():
                continue
            seed_meta = json.loads(oof_file.read_text())
            if not seed_meta.get("pooled"):
                continue
            pooled = seed_meta["pooled"]
            qwks.append(pooled["qwk"])
            accs.append(pooled["acc"])
            maes.append(pooled["mae"])
            ns.append(pooled["n"])
            probes.append(seed_meta.get("probe"))
            gates.append(seed_meta.get("gate_trace"))
        if not qwks:
            continue
        if cfg.get("unimodal"):
            fusion_label = "unimodal"
        else:
            fusion_label = cfg["fusion"]
        qwk_per_seed = []
        for seed_qwk in qwks:
            qwk_per_seed.append(round(seed_qwk, 4))
        if probes and probes[0]:
            probe_seed0 = probes[0]
        else:
            probe_seed0 = None
        rows.append({
            "config": name, "fusion": fusion_label,
            "skip": cfg.get("use_skip"), "mdrop": cfg.get("mdrop"),
            "loss": cfg.get("loss", "coral"), "unimodal": cfg.get("unimodal"),
            "shuffle": cfg.get("shuffle"),
            "qwk_mean": round(float(np.mean(qwks)), 4),
            "qwk_std": round(float(np.std(qwks)), 4),
            "qwk_per_seed": qwk_per_seed,
            "acc_mean": round(float(np.mean(accs)), 4),
            "mae_mean": round(float(np.mean(maes)), 4),
            "n_median": int(sorted(ns)[len(ns) // 2]),
            "probe_seed0": probe_seed0})
        # Final gate weights of the first fold (sorted by fold key) of seed 0.
        if gates and gates[0]:
            first_fold_trace = sorted(gates[0].items())[0][1]
            if first_fold_trace:
                rows[-1]["gate_alpha_final"] = first_fold_trace[-1]["gate"]
    rows.sort(key=summary_sort_key)
    if 'P2' in doc_key:
        pooled_over = 'over 5 folds'
    else:
        pooled_over = 'train/val'
    print(f"\n=== {doc_key} (pooled {pooled_over}"
          f", seeds {list(SEEDS)}) ===")
    print(f"{'config':28s} {'QWK mean±std':>14s} {'acc':>7s} {'mae':>6s} {'n':>5s}")
    for row in rows:
        print(f"{row['config']:28s} {row['qwk_mean']:+.4f}±{row['qwk_std']:.3f} "
              f"{row['acc_mean']:7.3f} {row['mae_mean']:6.3f} {row['n_median']:5d}")

    shuffle_rows = []
    for row in rows:
        if row.get("shuffle"):
            shuffle_rows.append(row)
    if shuffle_rows:
        absolute_qwks = []
        for row in shuffle_rows:
            absolute_qwks.append(abs(row["qwk_mean"]))
        max_abs_qwk = max(absolute_qwks)
        if max_abs_qwk <= SHUFFLE_TOL:
            verdict = 'PASS'
        else:
            verdict = 'FAIL'
        print(f"shuffle control: max |QWK| = {max_abs_qwk:.4f} "
              f"({verdict} ≤ {SHUFFLE_TOL})")
    unimodal_rows = []
    for row in rows:
        if row.get("unimodal"):
            unimodal_rows.append(row)
    if unimodal_rows:
        best = max(unimodal_rows, key=row_qwk_mean)
        print(f"best in-protocol unimodal: {best['config']} = {best['qwk_mean']:.4f}")

    if write_to.exists():
        doc = json.loads(write_to.read_text())
    else:
        doc = {}
    doc.setdefault(doc_key, {})["rows"] = rows
    write_to.write_text(json.dumps(doc, indent=2))
    return rows


def masked_eval(corpus, seeds=SEEDS):
    """H4: retrain gate/core skip1-drop0-soft; zero out each stream at eval.

    NOT USED in this scoped run. For each seed and config it trains on
    subject folds (only clips with every stream), then predicts the test
    fold normally ("none") and with each stream replaced by zeros
    ("mask_<stream>"). Returns {seed: {config name: result dict}}.
    """
    configs = ({'fusion': 'gate', 'use_skip': True, 'mdrop': False, 'loss': 'soft'},
               {'fusion': 'core', 'use_skip': True, 'mdrop': False, 'loss': 'soft'})
    all_present = []
    for stream_name in STREAMS:
        if corpus[f"feat_{stream_name}"].shape[0] > 0:
            all_present.append(stream_name)
    keep = np.ones(len(corpus["y"]), dtype=bool)
    for stream_name in all_present:
        keep &= corpus[f"avail_{stream_name}"]
    idx_all = np.flatnonzero(keep)
    ys = corpus["y"]
    per_seed = {}
    for seed in seeds:
        unique_subjects = sorted(set(corpus["subjects"][idx_all].tolist()))
        rng = np.random.default_rng(seed)
        rng.shuffle(unique_subjects)
        fold_of_subject = {}
        for position, subject in enumerate(unique_subjects):
            fold_of_subject[subject] = position % N_FOLDS
        fold_of_list = []
        for subject in corpus["subjects"][idx_all]:
            fold_of_list.append(fold_of_subject[subject])
        fold_of = np.array(fold_of_list)
        rows = {}
        for cfg in configs:
            name = cfg_name(cfg)
            condition_names = ["none"]
            for stream_name in all_present:
                condition_names.append(f"mask_{stream_name}")
            cond_pred = {}
            for condition in condition_names:
                cond_pred[condition] = np.full(len(corpus["y"]), np.nan)
            alpha_acc = {}
            for condition in cond_pred:
                alpha_acc[condition] = []
            dims = {}
            for stream_name in all_present:
                dims[stream_name] = int(corpus[f"feat_{stream_name}"].shape[1])
            for fold_number in range(N_FOLDS):
                test_index = idx_all[fold_of == fold_number]
                train_index = idx_all[~np.isin(idx_all, test_index)]
                if len(set(ys[train_index])) < 2 or len(test_index) == 0:
                    continue
                train_arrays = {}
                for stream_name in all_present:
                    train_arrays[stream_name] = corpus[f"feat_{stream_name}"][train_index]
                train_streams = to_torch(train_arrays)
                train_labels = torch.as_tensor(ys[train_index], device=device)
                train_soft = torch.as_tensor(corpus["soft"][train_index], device=device)
                test_arrays = {}
                for stream_name in all_present:
                    test_arrays[stream_name] = corpus[f"feat_{stream_name}"][test_index]
                test_streams = to_torch(test_arrays)
                test_labels = torch.as_tensor(ys[test_index], device=device)
                set_seed(int(f"{seed}{fold_number}"))
                model = GTAONetGrid(dims, fusion=cfg["fusion"],
                                    use_skip=cfg["use_skip"], mdrop=cfg["mdrop"])
                model = model.to(device)
                best, _, _ = fit_fold(model, train_streams, train_labels,
                                      test_streams, test_labels,
                                      subj_tr=corpus["subjects"][train_index],
                                      stream_list=all_present,
                                      loss=cfg["loss"], train_soft=train_soft)
                model.load_state_dict(best["state"])
                model.eval()
                with torch.no_grad():
                    logits, alpha = model(test_streams, apply_mdrop=False)
                    cond_pred["none"][test_index] = model.head.predict(logits).cpu().numpy()
                    if alpha is not None:
                        alpha_acc["none"].append(alpha.cpu().numpy())
                    for masked_stream in all_present:
                        masked_inputs = {}
                        for stream_name, tensor in test_streams.items():
                            if stream_name == masked_stream:
                                masked_inputs[stream_name] = torch.zeros_like(tensor)
                            else:
                                masked_inputs[stream_name] = tensor
                        logits_masked, alpha_masked = model(masked_inputs, apply_mdrop=False)
                        cond_pred[f"mask_{masked_stream}"][test_index] = (
                            model.head.predict(logits_masked).cpu().numpy())
                        if alpha_masked is not None:
                            alpha_acc[f"mask_{masked_stream}"].append(
                                alpha_masked.cpu().numpy())
            out = {"config": cfg, "seed": seed, "conditions": {}}
            for condition, predictions in cond_pred.items():
                predicted = np.isfinite(predictions)
                cond_qwk, cond_acc, cond_mae = pooled_scores(
                    ys[predicted].astype(int), predictions[predicted].astype(int))
                entry = {"qwk": cond_qwk,
                         "acc": cond_acc,
                         "mae": cond_mae,
                         "n": int(predicted.sum())}
                if alpha_acc[condition]:
                    all_alpha = np.concatenate(alpha_acc[condition], axis=0)
                    gate_alpha = []
                    for mean_weight in all_alpha.mean(axis=0):
                        gate_alpha.append(round(float(mean_weight), 4))
                    entry["gate_alpha"] = gate_alpha
                out["conditions"][condition] = entry
            rows[name] = out
            qwk_drops = []
            for condition, entry in out['conditions'].items():
                if condition.startswith('mask_'):
                    qwk_drops.append(out['conditions']['none']['qwk'] - entry['qwk'])
            print(f"  H4 {name} seed {seed}: "
                  f"none {out['conditions']['none']['qwk']:.3f} -> "
                  f"max masked drop "
                  f"{max(qwk_drops):.3f}")
        per_seed[seed] = rows
    return per_seed

# %% [markdown]
# ## Protocol P1 (official split, subject-filtered) — direct SOTA comparison

# %%
# =============================================================================
# SECTION 9. Protocol P1 (official split) - kept but NOT run
# =============================================================================
def run_p1(corpus):
    """Run the full factorial grid on the official split (not called by main)."""
    run_dir = RUNS / "P1_official"
    folds = official_split_folds(corpus)
    train_index, test_index = folds[0]
    print(f"P1 | train n={len(train_index)}  val n={len(test_index)}  "
          f"train subjects={len(set(corpus['subjects'][train_index].tolist()))}  "
          f"val subjects={len(set(corpus['subjects'][test_index].tolist()))}")
    configs = []
    for fusion, use_skip, mdrop, loss in product(FUSIONS, SKIPS, MDROPS, LOSSES):
        configs.append(dict({"fusion": fusion, "use_skip": use_skip, "mdrop": mdrop,
                             "loss": loss}))
    for stream_name in STREAMS:
        configs.append({"unimodal": stream_name, "loss": "coral"})
    for cfg in configs:
        for seed in SEEDS:
            oof_file = run_dir / cfg_name(cfg) / f"seed{seed}" / "oof.json"
            if oof_file.exists():
                continue
            seed_meta = run_protocol(corpus, folds, cfg, seed)
            if seed_meta is None:
                continue
            oof_file.parent.mkdir(parents=True, exist_ok=True)
            oof_file.write_text(json.dumps(seed_meta, indent=1))
    summarize(configs, "protocol_P1_official", run_dir, METRICS)

# %% [markdown]
# ## Protocol P2 (GroupKFold-5 by subject, full factorial) — DIPSER-style

# %%
# =============================================================================
# SECTION 10. Protocol P2 (5 subject-grouped folds x 3 seeds) - the real run
# =============================================================================
def run_p2(corpus):
    """Run the 6 scoped configs for every seed; skip cells whose oof.json exists.

    Writes engagenet_gt_grid/P2_groupkfold/<config>/seed<k>/oof.json and the
    summary under "protocol_P2_groupkfold" in METRICS. Returns the rows.
    """
    run_dir = RUNS / "P2_groupkfold"
    configs = [
        {"unimodal": "face", "loss": "coral"},
        {"unimodal": "facemesh", "loss": "coral"},
        {"unimodal": "head_pose", "loss": "coral"},
        {"fusion": "gate", "use_skip": True, "mdrop": False, "loss": "coral"},
        {"fusion": "core", "use_skip": True, "mdrop": False, "loss": "coral"},
        {"shuffle": True, "fusion": "core", "use_skip": True, "mdrop": True,
         "loss": "coral"},
    ]
    print(f"scoped P2 grid: {len(configs)} configs x {len(SEEDS)} seeds "
          f"x {N_FOLDS} folds "
          f"(est. ~{len(configs) * len(SEEDS) * 5 * 166 / 3600:.1f} h GPU, "
          f"+~0.3 h feature extraction; 5 h session cap)", flush=True)
    for seed in SEEDS:
        folds = subject_group_folds(corpus, seed)
        class_counts = np.bincount(corpus["y"])
        print(f"P2 seed {seed} | n={len(corpus['y'])} "
              f"subjects={len(set(corpus['subjects'].tolist()))} "
              f"class_dist={class_counts.tolist()}", flush=True)
        for cfg in configs:
            name = cfg_name(cfg)
            oof_file = run_dir / name / f"seed{seed}" / "oof.json"
            if oof_file.exists():
                continue
            seed_meta = run_protocol(corpus, folds, cfg, seed)
            if seed_meta is None:
                continue
            oof_file.parent.mkdir(parents=True, exist_ok=True)
            oof_file.write_text(json.dumps(seed_meta, indent=1))
    rows = summarize(configs, "protocol_P2_groupkfold", run_dir, METRICS)
    return rows

# %% [markdown]
# ## Main

# %%
# =============================================================================
# SECTION 11. Main
# =============================================================================
def analyze(corpus):
    """Print clips / subjects / class counts per Kaggle split and subject overlap."""
    splits = corpus["splits"]
    subjects = corpus["subjects"]
    print("\n== population ==")
    for split_name in ("Train", "Validation", "Test"):
        mask = splits == split_name
        print(f"  {split_name:10s} clips={int(mask.sum()):5d} "
              f"subjects={len(set(subjects[mask].tolist())):3d} "
              f"class={np.bincount(corpus['y'][mask], minlength=N_CLASSES).tolist()}")
    overlap = {}
    for split_a, split_b in [("Train", "Validation"), ("Train", "Test"),
                             ("Validation", "Test")]:
        subjects_a = set(subjects[splits == split_a].tolist())
        subjects_b = set(subjects[splits == split_b].tolist())
        overlap[split_a + "|" + split_b] = len(subjects_a & subjects_b)
    print(f"  subject overlap (Kaggle splits, NOT subject-disjoint): {overlap}")


def main():
    """Load data, mark the run as running, run P2, then mark it done."""
    t0 = time.time()
    corpus = load_corpus()
    METRICS.parent.mkdir(parents=True, exist_ok=True)
    if METRICS.exists():
        doc = json.loads(METRICS.read_text())
    else:
        doc = {}
    doc["status"] = "running"
    doc["env"] = {"device": device, "fast": FAST, "host": socket.gethostname()}
    METRICS.write_text(json.dumps(doc, indent=2))
    analyze(corpus)
    run_p2(corpus)
    doc = json.loads(METRICS.read_text())
    doc["status"] = "done"
    doc["finished_utc"] = time.strftime("%Y-%m-%d %H:%M:%S")
    doc["wall_s"] = round(time.time() - t0, 1)
    METRICS.write_text(json.dumps(doc, indent=2))
    print("total wall:", round(time.time() - t0, 1), "s")


if __name__ == "__main__":
    main()
