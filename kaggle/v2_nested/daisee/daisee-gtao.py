# === v2 NESTED: derived from v1 by tools/make_nested.py. Early stopping now uses an
# === inner subject split of the training fold (GTAO_NESTED=0 restores v1 behaviour).
"""gTAO-Net on DAiSEE: a 3-stream engagement benchmark (v2, "nested" edition).

READ THIS FIRST (beginner overview)
===================================

What does this script do?
-------------------------
It trains and evaluates a small neural network, gTAO-Net, that predicts how
engaged a student is in a 10-second webcam clip. The labels come from the
DAiSEE dataset: "Engagement" is an ordinal level 0, 1, 2 or 3 (0 = not
engaged, 3 = very engaged). "Ordinal" means the classes have an order, so
predicting 2 when the truth is 3 is a smaller mistake than predicting 0.

The same network, losses and hyper-parameters were used on two other datasets
(DIPSER and EngageNet); this file transfers that grid to DAiSEE unchanged so
the numbers are comparable with published DAiSEE results (ViBED-Net 73.43%,
Safa 67.32%, Abedi 67.4%, VisioPhysioENet 63.09%, all on the official split).

Each clip is turned into three numeric "streams" (one vector per clip):
  * face      - 2-D Tchebichef moments of the face crop, dimension 1536
  * facemesh  - 478 MediaPipe 3-D face landmarks per frame, dimension 8604
  * head_pose - head pitch / yaw / roll per frame (solvePnP), dimension 18
Six frames are sampled per clip (SEQ_K = 6). A 1-D Tchebichef transform over
those 6 frames summarises how each feature changes in time.

The data it expects (Kaggle inputs)
-----------------------------------
Attach the public DAiSEE mirrors (olgaparfenova/daisee, mahisharamesh/daisee).
The script searches everything under /kaggle/input for
  **/Labels/{Train,Validation,Test}Labels.csv   (columns ClipID, Engagement, ...)
  **/DataSet/{Train,Validation,Test}/<userID>/.../<clip>.avi
The <userID> folder is the "subject" (the person); the split (Train /
Validation / Test) comes from which labels CSV lists the clip.

The experiment grid
-------------------
A "config" is one combination of design choices:
  fusion  in {gate, core}         how the 3 stream embeddings are combined
  skip    in {ON, OFF}            an extra linear shortcut from raw inputs
  mdrop   in {ON, OFF}            modality dropout (randomly hide a stream)
  loss    in {coral, hybrid, soft}
That is 2 x 2 x 2 x 3 = 24 fusion configs, plus 3 single-stream ("unimodal")
baselines. Every config is trained with seeds 0, 1 and 2.

Three evaluation protocols, all "subject-disjoint" (no person appears in both
the training and the scoring data):
  * P1 - official split: train on Train, score on Validation (training users
    that also appear in Validation are removed).
  * P1-Test - for comparison with published papers, which score the official
    Test partition: the best single stream and the best fusion model OF P1
    (chosen on Validation, never on Test) are retrained on Train + Validation
    and score Test once. Runs last, after P2, so P1 and P2 are unaffected.
  * P2 - DIPSER-style: 5 folds grouped by subject, predictions of all 5
    held-out folds are pooled ("out-of-fold", OOF) and scored together.
    P2 also runs:
      - a shuffle control: labels randomly permuted; a sound pipeline must
        then score about 0 (|QWK| <= 0.10). Anything higher means leakage.
      - masked evaluation (hypothesis "H4"): at test time each stream is
        replaced by zeros to see how much the model relies on it.
The score is QWK (quadratic weighted kappa, 1 = perfect, 0 = chance), plus
accuracy and MAE (mean absolute error in class steps).

What "nested early stopping" means
----------------------------------
Training runs for up to EPOCHS epochs and we keep the weights of the epoch
that scored best on a validation set ("early stopping"). In v1 that
validation set was the SAME fold that was finally scored, which makes the
reported QWK optimistic. v2 ("nested") fixes this inside fit_fold:
  1. the training fold is split BY SUBJECT into inner-train (80%) and
     inner-validation (INNER_FRAC = 20% of the training subjects); the
     random draw is repeated (max 20 times) until both parts contain at
     least 2 classes;
  2. the model is trained on inner-train only;
  3. the best epoch is chosen on inner-validation;
  4. only then is the outer (scored) fold predicted, once.
So the scored fold never influences training or epoch selection. This
applies to P1, P2 and the masked evaluation. Every fold prints a line
"    inner split: train n=..., val n=..., val subjects=[...]".
In the per-run JSON, folds[*].best_val_qwk is therefore the INNER-validation
QWK; "pooled" is the outer-fold score. In nested mode each fold record also
gets "n_inner_train", "n_inner_val" and "inner_val_subjects" (sorted list),
after the other keys. GTAO_NESTED=0 restores v1 exactly (no extra keys).

(History: the first automatic v1 -> v2 patch put this block in the wrong
function, official_split_folds, which crashed with NameError under
GTAO_NESTED=1 and never performed the inner split. That has been fixed.)

How to run it
-------------
On Kaggle: push it as a "script" kernel with GPU and INTERNET ON (it
pip-installs mediapipe 0.10.14 and protobuf 4.25.3 at start). The first run
decodes all videos (hours) and caches the features to daisee_clips_v1.npz.

Locally (quick smoke test on synthetic random data, CPU is fine):
    GRID_FAST=1 python daisee-gtao.py
  GRID_FAST=1   -> 8 epochs instead of 40; at most 400 clips. If no DAiSEE
                   videos are found, a synthetic 200-clip corpus is
                   generated instead.
  GTAO_NESTED=0 -> v1 behaviour (early stopping on the scored fold).
  DAISEE_ROOT   -> optional folder containing a local copy of DAiSEE.
                   Careful: if it is unset, the CURRENT folder is searched
                   recursively for *Labels.csv and *.avi files.
On Windows PowerShell: $env:GRID_FAST="1"; python daisee-gtao.py
(On Windows "/tmp/daisee_gt_out" means the folder tmp/daisee_gt_out at the root of the current drive.)
If a previous run's daisee_clips_v1.npz or oof.json files are still in the
output folder they are reused / skipped, so delete the folder for a fresh run.
Resuming: attach an earlier version's output to the kernel as an input (or,
locally, set GTAO_RESUME_FROM=<its output folder>); its cache, metrics and
oof.json files are copied in first, so only unfinished work is run.

What it writes (to /kaggle/working on Kaggle, else /tmp/daisee_gt_out)
----------------------------------------------------------------------
  daisee_clips_v1.npz       cached per-clip stream features (reused next run)
  daisee_gt_metrics.json    summary: "status", "env", "protocol_P1_official",
                            "protocol_P2_groupkfold", "h4_masked_eval_P2",
                            "protocol_P1_test", "finished_utc", "wall_s"
  daisee_gt_grid/P1_official/<config>/seed<k>/oof.json    per-run details
  daisee_gt_grid/P2_groupkfold/<config>/seed<k>/oof.json
  daisee_gt_grid/P1_test/<config>/seed<k>/oof.json        (2 configs only)
Downstream tools (tools/pull_v2.sh, paper/analysis/stats.py, tables.py) read
these files, so their names and JSON keys must not change.

Notes on this readable version
------------------------------
The original file also contained an older EngageNet crop loader (find_crops,
load_index, SUBJECT_RE and first versions of MeshExtractor, encode_clip and
synth_corpus). Those definitions were either never called or were silently
replaced by the later definitions of the same name, so they have been removed.
Unused imports (collections, re, tempfile, PIL.Image, sklearn) were removed too.
With GTAO_NESTED=0 every number produced is bit-identical to the published
v1 script (same order of all random-number calls, same model construction
order, same floating-point operations).
"""

# %%
# =============================================================================
# SECTION 1. Imports
# =============================================================================
import csv
import json
import math
import os
import shutil
import socket
import time
from itertools import product
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import cv2  # OpenCV: video decoding, colour conversion, solvePnP
except ImportError:  # local smoke test without OpenCV
    cv2 = None

# %%
# =============================================================================
# SECTION 2. Kaggle environment set-up (pip install + protobuf shim)
# =============================================================================
# On Kaggle we install the exact library versions the experiment needs.
# mediapipe 0.10.14 expects a protobuf module file "runtime_version.py" that
# protobuf 4.25.3 does not ship, so we write a tiny stand-in file ("shim").
ON_KAGGLE = os.path.isdir("/kaggle/working")
if ON_KAGGLE:
    os.system("pip -q install mpmath scikit-learn opencv-python-headless 'protobuf==4.25.3' 'mediapipe==0.10.14'")
    import pathlib as _pl
    import google.protobuf as _gp
    protobuf_package_dir = _pl.Path(_gp.__file__).parent
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
    print("PB_VER", _gp.__version__)
    print("RV_EXISTS_AFTER_SHIM", runtime_version_file.exists())
    try:
        import mediapipe as _mp
        print("MP_OK_TOP", _mp.__version__)
    except Exception as _e:
        print("MP_FAIL_TOP", repr(_e)[:400])
        raise SystemExit(1) from _e

# These two libraries are imported only AFTER the pip install above, so that on
# Kaggle the freshly installed versions are the ones that get loaded.
import mpmath  # noqa: E402  (arbitrary-precision maths for the Tchebichef basis)
from sklearn.metrics import cohen_kappa_score  # noqa: E402

# %%
# =============================================================================
# SECTION 3. Configuration (paths, hyper-parameters, experiment grid)
# =============================================================================
if ON_KAGGLE:
    OUT = Path("/kaggle/working")
else:
    OUT = Path("/tmp/daisee_gt_out")
OUT.mkdir(parents=True, exist_ok=True)
METRICS = OUT / "daisee_gt_metrics.json"       # summary JSON (read by paper tools)
CACHE = OUT / "daisee_clips_v1.npz"            # per-clip stream tensors (reused)
RUNS = OUT / "daisee_gt_grid"                  # one oof.json per config and seed

FAST = os.environ.get("GRID_FAST", "") == "1"  # quick local smoke mode
SEQ_K = 6                                      # frames per clip = 1-D Tchebichef order in time
FACE_SIZE = 64                                 # face crop is resized to 64 x 64 pixels
FACE_K = 16                                    # 2-D Tchebichef order -> 16 x 16 = 256 moments
N_CLASSES = 4                                  # DAiSEE Engagement 0..3 (ordinal)
EMB = 64                                       # size of each stream embedding
HID = 128                                      # hidden width of each stream tower
LR = 3e-3                                      # Adam learning rate
if FAST:
    EPOCHS = 8
else:
    EPOCHS = 40
BATCH = 32                                     # mini-batch size
PATIENCE = 8                                   # stop after 8 epochs without improvement
# v2: nested early stopping. Epoch selection uses an inner validation split of
# the TRAINING-fold subjects; the scored outer fold never influences training.
NESTED = os.environ.get("GTAO_NESTED", "1") == "1"
INNER_FRAC = 0.2                               # share of training subjects held out
MIN_EPOCHS = 5                                 # never stop before this epoch
N_FOLDS = 5                                    # P2: number of subject-grouped folds
SEEDS = (0, 1, 2)
MDROP_P = 0.2                                  # modality dropout probability per stream

STREAMS = ("face", "facemesh", "head_pose")
FUSIONS = ("gate", "core")
SKIPS = (True, False)
MDROPS = (True, False)
LOSSES = ("coral", "hybrid", "soft")

SHUFFLE_TOL = 0.10                             # shuffle control passes if |QWK| <= this
if FAST:
    MAX_CLIPS = 400
else:
    MAX_CLIPS = None

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
print("device:", device, "| FAST:", FAST)

# %%
# =============================================================================
# SECTION 4. Tchebichef moments (identical to gtao_grid / pipeline/tchebichef.py)
# =============================================================================
# Discrete Tchebichef polynomials t_0..t_{K-1} are orthogonal polynomials on the
# integer grid x = 0..N-1. Projecting a signal onto the first K of them gives K
# "moments" that summarise its shape (like a Fourier transform, but with
# polynomials). They are built with the three-term recurrence
#     t_0(x) = 1
#     t_1(x) = 2x - N + 1
#     n t_n(x) = (2n-1)(2x-N+1) t_{n-1}(x) - (n-1)(N^2-(n-1)^2) t_{n-2}(x)
# and normalised by sqrt(rho(n, N)), where rho is the squared norm.
# mpmath with 50 digits is used because the raw values grow very large.

def _rho(n, N):
    """Squared norm of the n-th Tchebichef polynomial on N points.

    rho(n, N) = N (N^2 - 1^2)(N^2 - 2^2)...(N^2 - n^2) / (2n + 1)
    Returns an mpmath number (high precision).
    """
    r = mpmath.mpf(N)
    for k in range(1, n + 1):
        r *= N * N - k * k
    return r / (2 * n + 1)


def tchebichef_basis(N, K):
    """Build the orthonormal Tchebichef basis matrix.

    Inputs:  N = signal length, K = number of polynomials (moments) kept.
    Output:  float32 numpy array B of shape (N, K); column n is t_n / ||t_n||.
    Multiplying a length-N signal by B gives its K Tchebichef moments.
    """
    with mpmath.workdps(50):
        # t[n][x] holds polynomial n evaluated at point x (K rows of N values).
        t = []
        for _ in range(K):
            t.append([mpmath.mpf(0)] * N)
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


_BASIS = {}  # cache: (N, K) -> basis matrix, so each basis is computed only once


def basis(N, K):
    """Return the (N, K) Tchebichef basis, computing it on first use only."""
    key = (N, K)
    if key not in _BASIS:
        _BASIS[key] = tchebichef_basis(N, K)
    return _BASIS[key]


def transform_2d(img, K):
    """2-D Tchebichef moments of a square image.

    img: (S, S) array.  Returns (K, K) array  B^T @ img @ B  where B is (S, K).
    """
    B = basis(img.shape[-1], K)
    return (B.T @ img) @ B


def transform_1d(x, K):
    """1-D Tchebichef moments along the LAST axis.

    x: (..., N) array.  Returns (..., K) array  x @ B  where B is (N, K).
    """
    B = basis(x.shape[-1], K)
    return x @ B


def pad_time(arr, k):
    """Make sure the time axis (axis 0) has at least k rows.

    If arr has fewer than k frames, the last frame is repeated until there are
    k. arr: (T, D) array; returns (max(T, k), D). Raises if T == 0.
    """
    if arr.shape[0] >= k:
        return arr
    if arr.shape[0] == 0:
        raise ValueError("empty time axis")
    n_missing = k - arr.shape[0]
    repeated_last_frame = np.repeat(arr[-1:], n_missing, axis=0)
    return np.concatenate([arr, repeated_last_frame], axis=0)


def temporal_moments(per_frame_array):
    """Turn a (T, D) per-frame array into one clip vector of length D * SEQ_K.

    Steps: pad time to SEQ_K frames -> move time to the last axis (D, SEQ_K)
    -> 1-D Tchebichef over time (D, SEQ_K) -> flatten to (D * SEQ_K,).
    E.g. face: 256 moments x 6 = 1536; facemesh: 478*3 = 1434 x 6 = 8604;
    head_pose: 3 x 6 = 18.
    """
    padded = pad_time(per_frame_array, SEQ_K)          # (T>=SEQ_K, D)
    time_last = np.moveaxis(padded, 0, -1)             # (D, T)
    rows = time_last.reshape(-1, SEQ_K)                # (D, SEQ_K)
    return transform_1d(rows, SEQ_K).ravel()           # (D * SEQ_K,)

# %%
# =============================================================================
# SECTION 5. DAiSEE loader: .avi clips -> per-clip stream tensors (cached npz)
# =============================================================================
# Front-end identical to EngageNet/DIPSER: for each sampled frame run MediaPipe
# FaceMesh (478 refined landmarks) -> face crop (Tchebichef moments) + landmark
# mesh + solvePnP head pose. Expected layout (verified for olgaparfenova/daisee):
#   **/Labels/{Train,Validation,Test}Labels.csv  (ClipID, Boredom, Engagement, ...)
#   **/DataSet/{Train,Validation,Test}/<userID>/.../<clip>.avi
# The userID folder is the subject group; the split comes from the labels CSV.

def dataset_roots():
    """Folders to search for DAiSEE files.

    On Kaggle: /kaggle/input. Locally: the folder in env var DAISEE_ROOT
    (an empty value gives Path(""), i.e. the current folder; callers use
    rglob only after checking is_dir()).
    """
    if ON_KAGGLE:
        return [Path("/kaggle/input")]
    return [Path(os.environ.get("DAISEE_ROOT", ""))]


def split_from_label_filename(file_name):
    """Map a labels CSV file name to its split: 'Train', 'Validation', 'Test' or None."""
    lower_name = file_name.lower()
    if "train" in lower_name:
        return "Train"
    if "valid" in lower_name:
        return "Validation"
    if "test" in lower_name:
        return "Test"
    return None


def read_daisee_labels():
    """Read every *Labels.csv file found under the dataset roots.

    Returns a dict  clip_id (file name without extension) -> (engagement 0..3, split).
    Rows that cannot be parsed are skipped silently; files without a ClipID
    or Engagement column are skipped with a message.
    """
    labels = {}
    roots = dataset_roots()
    found = 0
    for root in roots:
        if not root.is_dir():
            continue
        for label_file in sorted(root.rglob("*Labels.csv")):
            split = split_from_label_filename(label_file.name)
            if split is None:
                continue
            found += 1
            try:
                with open(label_file, newline="") as handle:
                    reader = csv.DictReader(handle)
                    # Map "normalised column name" -> real column name, so that
                    # "ClipID", " clipid " etc. are all found.
                    column_names = reader.fieldnames
                    if not column_names:
                        column_names = []
                    columns = {}
                    for column in column_names:
                        columns[column.lower().strip()] = column
                    clip_id_column = columns.get("clipid")
                    if not clip_id_column:
                        clip_id_column = columns.get("clip id")
                    engagement_column = columns.get("engagement")
                    if not clip_id_column or not engagement_column:
                        print("  skip labels (no clipid/engagement col):", label_file.name)
                        continue
                    for row in reader:
                        try:
                            clip_id = Path(str(row[clip_id_column])).stem
                            labels[clip_id] = (int(float(row[engagement_column])), split)
                        except Exception:
                            continue
            except Exception as error:
                print("  label read skip", label_file, repr(error))
    print(f"labels: {len(labels)} clips from {found} CSV(s)")
    return labels


def find_videos():
    """Return a list of all .avi files under the dataset roots (sorted per root)."""
    roots = dataset_roots()
    videos = []
    for root in roots:
        if root.is_dir():
            videos += sorted(root.rglob("*.avi"))
    return videos


def subject_of(avi):
    """Return the subject (user) id of a clip from its path.

    Layout .../<split>/<userID>/<clipID>/<clip>.avi  -> parts[-3] is <userID>.
    If instead parts[-3] is the split name (layout .../<split>/<userID>/<clip>.avi)
    then the user id is parts[-2].
    """
    parts = avi.parts
    candidate = parts[-3].lower()
    if candidate in ("train", "validation", "test"):
        return parts[-2]
    return parts[-3]


class MeshExtractor:
    """MediaPipe FaceMesh (478 landmarks, refine_landmarks) + solvePnP head pose + face box.

    If mediapipe cannot be loaded, self.fm stays None and detect() returns
    (None, None, None) for every frame.
    """

    def __init__(self):
        """Create the MediaPipe FaceMesh detector (one face, still-image mode)."""
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

    def detect(self, rgb):
        """Find the face in one RGB frame.

        Input:  rgb, (h, w, 3) uint8 image.
        Output: (mesh, pose, bbox)
            mesh: (478, 3) float32 landmarks in pixel units (x, y) + relative depth z
            pose: (3,) float32 [pitch, yaw, roll] in degrees, or None
            bbox: (x0, y0, x1, y1) face box enlarged by 15%
        or (None, None, None) if no face is found. A second attempt is made with
        a 10% white border, which helps when the face touches the image edge.
        """
        if self.fm is None:
            return None, None, None
        h, w = rgb.shape[:2]
        for attempt in range(2):
            if attempt == 0:
                pad = 0
            else:
                pad = int(0.1 * max(h, w))
            if pad:
                image = cv2.copyMakeBorder(rgb, pad, pad, pad, pad,
                                           cv2.BORDER_CONSTANT, value=(255, 255, 255))
            else:
                image = rgb
            result = self.fm.process(image)
            if result.multi_face_landmarks:
                face_landmarks = result.multi_face_landmarks[0]
                # MediaPipe gives x, y as fractions of the image size: convert to pixels.
                point_list = []
                for landmark in face_landmarks.landmark:
                    point_list.append((landmark.x * image.shape[1],
                                       landmark.y * image.shape[0],
                                       landmark.z))
                pts = np.array(point_list, dtype=np.float32)        # (478, 3)
                if pad:
                    pts[:, :2] -= pad          # undo the border shift
                pose = self._head_pose(pts.astype(np.float64), w, h)
                xs, ys = pts[:, 0], pts[:, 1]
                margin = 0.15
                x0 = max(0, int(xs.min() - margin * (xs.max() - xs.min())))
                y0 = max(0, int(ys.min() - margin * (ys.max() - ys.min())))
                x1 = min(w - 1, int(xs.max() + margin * (xs.max() - xs.min())))
                y1 = min(h - 1, int(ys.max() + margin * (ys.max() - ys.min())))
                return pts, pose, (x0, y0, x1, y1)
        return None, None, None

    @staticmethod
    def _head_pose(pts, w, h):
        """Estimate head rotation from 6 landmarks with OpenCV solvePnP.

        Matches 6 landmarks (nose tip, chin, eye corners, mouth corners) to a
        generic 3-D face model, using a simple pinhole camera (focal length =
        max(w, h), centre of image). The rotation matrix R is converted to
        Euler angles. Returns float32 [pitch, yaw, roll] in degrees, or None.
        """
        model_points_3d = np.array([
            [0.0, 0.0, 0.0], [0.0, -63.6, -12.5],
            [-43.3, 32.7, -26.0], [43.3, 32.7, -26.0],
            [-28.9, -28.9, -24.1], [28.9, -28.9, -24.1]], dtype=np.float64)
        landmark_ids = [1, 152, 33, 263, 61, 291]
        try:
            image_point_list = []
            for x, y, _ in pts[landmark_ids]:
                image_point_list.append([x, y])
            image_points_2d = np.array(image_point_list, dtype=np.float64)   # (6, 2)
            focal = max(w, h)
            camera_matrix = np.array([[focal, 0, w / 2], [0, focal, h / 2], [0, 0, 1]],
                                     dtype=np.float64)
            ok, rotation_vector, _ = cv2.solvePnP(model_points_3d, image_points_2d,
                                                  camera_matrix, None,
                                                  flags=cv2.SOLVEPNP_ITERATIVE)
            if not ok:
                return None
            R, _ = cv2.Rodrigues(rotation_vector)          # 3x3 rotation matrix
            sy = math.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
            pitch = math.atan2(-R[2, 1], R[2, 2])
            yaw = math.atan2(R[2, 0], sy)
            roll = math.atan2(-R[1, 0], R[0, 0])
            return np.array([math.degrees(pitch), math.degrees(yaw),
                             math.degrees(roll)], dtype=np.float32)
        except Exception:
            return None


def frame_streams(frame_bgr, ext):
    """Compute the per-frame features of one video frame.

    Input: frame_bgr (h, w, 3) uint8 as read by OpenCV; ext = MeshExtractor or None.
    Output: dict with
        "face":      (256,) float32  2-D Tchebichef moments of the 64x64 grey face crop
        "facemesh":  (478, 3) array or None
        "head_pose": (3,) array or None
    If no face is detected, a centred square covering 80% of the shorter side
    is used as the "face" crop.
    """
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    if ext is not None:
        mesh, pose, bbox = ext.detect(rgb)
    else:
        mesh, pose, bbox = (None, None, None)
    if bbox is None:
        h, w = rgb.shape[:2]
        side = int(0.8 * min(h, w))
        y0 = (h - side) // 2
        x0 = (w - side) // 2
        bbox = (x0, y0, x0 + side, y0 + side)
    x0, y0, x1, y1 = bbox
    crop = cv2.cvtColor(rgb[y0:y1, x0:x1], cv2.COLOR_RGB2GRAY)
    crop = cv2.resize(crop, (FACE_SIZE, FACE_SIZE)).astype(np.float32) / 255.0   # (64, 64)
    face = transform_2d(crop, FACE_K).ravel().astype(np.float32)                 # (256,)
    return {"face": face, "facemesh": mesh, "head_pose": pose}


def encode_clip(video, clip_id, label, split, subject, ext):
    """Decode one .avi clip and turn it into a record of clip-level features.

    SEQ_K frames are sampled uniformly over the clip, each frame is passed
    through frame_streams, and every stream is summarised over time with
    temporal_moments. facemesh / head_pose use only the frames where a face
    was found (and are left out if none was found).
    Returns a dict with key, subject, split, label, soft (one-hot), face and
    optionally facemesh / head_pose; or None if the video cannot be read.
    """
    capture = cv2.VideoCapture(str(video))
    # (The original wrote int(...) or 0; "or 0" never changes an int, so it is omitted.)
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        capture.release()
        return None
    frame_indices = np.linspace(0, total - 1, SEQ_K).astype(int)
    frames = []
    for frame_index in frame_indices:
        capture.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
        ok, frame = capture.read()
        if ok:
            frames.append(frame)
    capture.release()
    if not frames:
        return None
    per_frame = []
    for frame in frames:
        per_frame.append(frame_streams(frame, ext))

    face_list = []
    for frame_features in per_frame:
        face_list.append(frame_features["face"])
    faces = np.stack(face_list)                                    # (T, 256)
    feats = {"face": temporal_moments(faces)}                      # (1536,)
    for stream_key in ("facemesh", "head_pose"):
        arrays = []
        for frame_features in per_frame:
            if frame_features[stream_key] is not None:
                arrays.append(frame_features[stream_key])
        if not arrays:
            continue
        stacked = np.stack(arrays)                                 # (T', 478, 3) or (T', 3)
        feats[stream_key] = temporal_moments(stacked)
    soft = np.zeros(N_CLASSES, dtype=np.float32)
    soft[label] = 1.0
    record = {"key": f"{split}/{subject}/{clip_id}", "subject": subject,
              "split": split, "label": label, "soft": soft}
    for stream_key, value in feats.items():
        record[stream_key] = value
    return record


def records_to_basic_arrays(records):
    """Collect the per-clip metadata of a list of records into numpy arrays.

    Returns (keys, subjects, splits, y, soft):
        keys/subjects/splits: (n,) string arrays
        y:    (n,) int64 labels 0..3
        soft: (n, N_CLASSES) float32 one-hot labels
    """
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
    return keys, subjects, splits, y, soft


def load_daisee():
    """Encode every labelled clip once and cache the clip-level tensors to CACHE.

    If CACHE already exists it is simply loaded. If no videos are found, a
    synthetic corpus is created instead (local smoke test).
    Returns a numpy NpzFile ("corpus") with arrays:
        keys, subjects, splits, y, soft,
        avail_<stream> (n,) bool   - was the stream found for this clip?
        feat_<stream>  (n, D) float32 (NaN rows where missing; (0, 0) if absent everywhere)
    """
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

    is_labelled_list = []
    for video in videos:
        is_labelled_list.append(video.stem in labels)
    mask = np.asarray(is_labelled_list)
    labelled = []
    for video, is_labelled in zip(videos, mask):
        if is_labelled:
            labelled.append(video)
    print(f"videos: {len(videos)} total, {len(labelled)} labelled "
          f"({100 * mask.mean():.1f}%)")

    if cv2 is not None:
        ext = MeshExtractor()
    else:
        ext = None
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
        record = encode_clip(video, clip_id, label, split, subject, ext)
        if record is not None:
            records.append(record)
        if (n + 1) % 1000 == 0:
            print(f"  {n + 1}/{len(labelled)} clips encoded "
                  f"[{time.time() - t0:.0f}s]", flush=True)

    if not records:
        raise SystemExit("no clips encoded successfully")
    keys, subjects, splits, y, soft = records_to_basic_arrays(records)

    # avail[s][i] is True when clip i has stream s.
    avail = {}
    for s in STREAMS:
        present_list = []
        for record in records:
            present_list.append(record.get(s) is not None)
        avail[s] = np.asarray(present_list, dtype=bool)

    # feats[s] is an (n, D) matrix with NaN rows for clips missing stream s,
    # or None if no clip has stream s at all.
    feats = {}
    for s in STREAMS:
        feats[s] = None
    for s in STREAMS:
        idx = []
        for i, record in enumerate(records):
            if record.get(s) is not None:
                idx.append(i)
        if idx:
            vectors = []
            for i in idx:
                vectors.append(records[i][s])
            arr = np.asarray(vectors, dtype=np.float32)
            full = np.full((len(records), arr.shape[1]), np.nan, dtype=np.float32)
            full[idx] = arr
            feats[s] = full
        n_records_or_one = len(records)
        if not n_records_or_one:
            n_records_or_one = 1
        print(f"stream {s}: present in {len(idx)}/{len(records)} "
              f"({100 * len(idx) / n_records_or_one:.1f}%)")

    empty2d = np.zeros((0, 0), dtype=np.float32)
    feat_arrays = {}
    for s in STREAMS:
        if feats[s] is not None:
            feat_arrays[s] = feats[s]
        else:
            feat_arrays[s] = empty2d
    np.savez(CACHE, keys=keys, subjects=subjects, splits=splits, y=y, soft=soft,
             avail_face=avail["face"], avail_facemesh=avail["facemesh"],
             avail_head_pose=avail["head_pose"],
             feat_face=feat_arrays["face"],
             feat_facemesh=feat_arrays["facemesh"],
             feat_head_pose=feat_arrays["head_pose"])
    print(f"corpus: {len(records)} clips, {len(set(subjects.tolist()))} subjects, "
          f"{time.time() - t0:.0f}s")
    return np.load(CACHE, allow_pickle=True)


def synth_corpus():
    """Local GRID_FAST smoke: 200 random clips from 20 fake subjects.

    Subjects 0..3 form the "Validation" split, the rest "Train". Labels cycle
    0,1,2,3 and each stream gets a small label-dependent shift so that there
    is something to learn. Uses its own random generator (seed 0), so the
    corpus is the same on every run. Writes CACHE and returns it loaded.
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
        # Random-number order matters for reproducibility: face, mesh, head pose.
        face = rng.normal(size=(1536,)).astype(np.float32)
        face += 0.1 * label
        mesh = rng.normal(size=(8604,)).astype(np.float32) + 0.05 * label
        hp = rng.normal(size=(18,)).astype(np.float32) + 0.2 * label
        soft = np.zeros(N_CLASSES, dtype=np.float32)
        soft[label] = 1.0
        records.append({"key": f"{split}/{subj}/{i}", "subject": subj,
                        "split": split, "label": label, "soft": soft,
                        "face": face, "facemesh": mesh, "head_pose": hp})
    keys, subjects, splits, y, soft = records_to_basic_arrays(records)
    face_list = []
    mesh_list = []
    head_pose_list = []
    for record in records:
        face_list.append(record["face"])
        mesh_list.append(record["facemesh"])
        head_pose_list.append(record["head_pose"])
    np.savez(CACHE, keys=keys, subjects=subjects, splits=splits, y=y, soft=soft,
             avail_face=np.ones(n, dtype=bool),
             avail_facemesh=np.ones(n, dtype=bool),
             avail_head_pose=np.ones(n, dtype=bool),
             feat_face=np.stack(face_list),
             feat_facemesh=np.stack(mesh_list),
             feat_head_pose=np.stack(head_pose_list))
    return np.load(CACHE, allow_pickle=True)

# %%
# =============================================================================
# SECTION 6. The gTAO-Net model (copy of gtao_grid.py)
# =============================================================================
# Shapes used below: B = batch size, S = number of streams, E = EMB = 64.
#
#   stream inputs  {name: (B, D_name)}
#        |  one StreamTower per stream
#        v
#   z  (B, S, E)  --(optional modality dropout)--> fusion (gate or core) -> (B, E)
#        + optional GlobalSkip (linear map of the raw inputs)             -> (B, E)
#        v
#   CORAL head -> (B, N_CLASSES - 1) logits

class StreamTower(nn.Module):
    """Small MLP that maps one stream (B, in_dim) to an embedding (B, EMB).

    Linear(in_dim -> HID) -> GELU -> Linear(HID -> EMB) -> LayerNorm.
    """

    def __init__(self, in_dim, hidden=HID, emb=EMB):
        """Create the layers (parameter initialisation happens here)."""
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, hidden), nn.GELU(),
                                 nn.Linear(hidden, emb), nn.LayerNorm(emb))

    def forward(self, x):
        """x: (B, in_dim) -> (B, emb)."""
        return self.net(x)


class GatedCrossModalFusion(nn.Module):
    """"gate" fusion: cross-stream attention, then a learned softmax gate.

    1. Attention between streams: attn = softmax(Q K^T / sqrt(E)), shape (B, S, S),
       z2 = attn @ z mixes information across streams, shape (B, S, E).
    2. Gate: a small MLP looks at all streams and outputs one weight per
       stream, alpha (B, S), which sums to 1 (softmax).
    3. Output: weighted sum of the streams, sum_s alpha_s * z2_s, shape (B, E).
    The gate weights alpha are returned too, so we can see which stream the
    model relies on.
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
        q = self.proj_q(z)                                              # (B, S, E)
        k = self.proj_k(z)                                              # (B, S, E)
        attn = torch.softmax(q @ k.transpose(1, 2) / math.sqrt(z.shape[-1]), dim=-1)  # (B, S, S)
        z2 = attn @ z                                                   # (B, S, E)
        alpha = torch.softmax(self.gate_net(z2.reshape(z2.shape[0], -1)), dim=-1)     # (B, S)
        fused = (alpha.unsqueeze(-1) * z2).sum(dim=1)                   # (B, E)
        return fused, alpha


class CoreAttentionFusion(nn.Module):
    """"core" fusion: the same cross-stream attention, then a plain average.

    No gate, so the second return value (gate weights) is None.
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
        attn = torch.softmax(q @ k.transpose(1, 2) / math.sqrt(z.shape[-1]), dim=-1)
        z2 = attn @ z
        return z2.mean(dim=1), None


class GlobalSkip(nn.Module):
    """Skip connection: one bias-free linear map per raw stream, summed.

    Output = sum over streams of W_s x_s, shape (B, E). It lets raw input
    information reach the head without passing through the towers.
    """

    def __init__(self, stream_dims, emb=EMB):
        """stream_dims: {stream name: input dimension} (iteration order is kept)."""
        super().__init__()
        projections = {}
        for stream_name, dim in stream_dims.items():
            projections[stream_name] = nn.Linear(dim, emb, bias=False)
        self.proj = nn.ModuleDict(projections)

    def forward(self, streams):
        """streams: {name: (B, D_name)} -> (B, E). Summed in the dict's order."""
        projected = []
        for stream_name, x in streams.items():
            projected.append(self.proj[stream_name](x))
        return torch.stack(projected).sum(dim=0)


class CORALHead(nn.Module):
    """CORAL ordinal head (COnsistent RAnk Logits).

    Instead of 4 independent class scores, it predicts K-1 = 3 binary
    questions "is the label > k?" for k = 0, 1, 2. All questions share the
    same weight vector w and differ only by a bias b_k, so
        logit_k = w . x + b_k            (shape (B, K-1))
        P(y > k) = sigmoid(logit_k)
    Class probabilities follow from differences:
        P(y = k) = P(y > k-1) - P(y > k)   with P(y > -1) = 1, P(y > K-1) = 0.
    """

    def __init__(self, emb=EMB, num_classes=N_CLASSES):
        """Create the shared weight w and the K-1 biases (start at 0.5 .. -0.5)."""
        super().__init__()
        self.d = num_classes - 1
        self.w = nn.Linear(emb, 1)
        self.b = nn.Parameter(torch.linspace(0.5, -0.5, self.d))

    def forward(self, x):
        """x: (B, E) -> logits (B, K-1)."""
        return self.w(x) + self.b

    def probs(self, logits):
        """logits (B, K-1) -> class probabilities (B, K), each row sums to 1."""
        p_greater = torch.sigmoid(logits)                         # P(y > k), (B, K-1)
        extended = torch.cat([torch.ones_like(p_greater[:, :1]), p_greater,
                              torch.zeros_like(p_greater[:, :1])], dim=1)   # (B, K+1)
        p = (extended[:, :-1] - extended[:, 1:]).clamp(min=0)    # (B, K)
        return p / p.sum(dim=1, keepdim=True)

    def loss(self, logits, y):
        """Standard CORAL loss: binary cross-entropy on the K-1 "y > k" targets."""
        k = torch.arange(self.d, device=logits.device)[None, :]  # (1, K-1)
        targets = (y[:, None] > k).float()                       # (B, K-1)
        return F.binary_cross_entropy_with_logits(logits, targets)

    def predict(self, logits):
        """logits (B, K-1) -> predicted class index (B,), the most probable class."""
        return self.probs(logits).argmax(dim=1)


def head_loss(head, logits, y, soft, loss="coral"):
    """Training loss for the three loss variants of the grid.

    "coral":  CORAL binary cross-entropy (see CORALHead.loss).
    "soft":   the same BCE but with targets 1 - cumsum(soft) built from the
              (one-hot) soft labels; for one-hot labels this equals "coral".
    "hybrid": cross-entropy on class probabilities
              + 0.5 * CORAL BCE
              + 0.1 * cross-entropy against a smoothed target that puts 0.7
                on the true class and 0.15 on each neighbouring class
                (then renormalised).
    Returns a scalar tensor.
    """
    if loss == "coral":
        return head.loss(logits, y)
    if loss == "soft":
        cumulative = torch.cumsum(soft, dim=1)[:, :head.d]        # P(y <= k), (B, K-1)
        return F.binary_cross_entropy_with_logits(logits, 1.0 - cumulative)
    # ----- hybrid -----
    p = head.probs(logits)                                        # (B, K)
    ce = -torch.log(p.gather(1, y[:, None]).squeeze(1) + 1e-8).mean()
    bce = head.loss(logits, y)
    q = torch.zeros_like(p)
    q.scatter_(1, y[:, None], 0.7)                                # 0.7 on the true class
    for offset in (-1, 1):                                        # 0.15 on each neighbour
        neighbour = y + offset
        rows = torch.nonzero((neighbour >= 0) & (neighbour < N_CLASSES)).squeeze(1)
        if rows.numel():
            q[rows, neighbour[rows].clamp(0, N_CLASSES - 1)] += 0.15
    q = q / q.sum(dim=1, keepdim=True)
    smooth = -(q * torch.log(p + 1e-8)).sum(dim=1).mean()
    return ce + 0.5 * bce + 0.1 * smooth


class GTAONetGrid(nn.Module):
    """The full gTAO-Net used in the grid.

    stream_dims: {stream name: input dimension}
    fusion:      "gate" or "core"
    use_skip:    add the GlobalSkip shortcut
    mdrop:       use modality dropout while training
    Construction order (towers, fusion, skip, head) fixes the random
    parameter initialisation, so it must not be changed.
    """

    def __init__(self, stream_dims, fusion="gate", use_skip=True, mdrop=False,
                 emb=EMB):
        """Build the towers (in sorted stream order), fusion, skip and head."""
        super().__init__()
        self.streams = sorted(stream_dims)
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
        """streams: {name: (B, D_name)} -> (logits (B, K-1), gate alpha (B, S) or None).

        Modality dropout: while training with mdrop=True (or if apply_mdrop is
        truthy), each stream embedding of each sample is zeroed with
        probability MDROP_P, so the model learns not to depend on one stream.
        """
        embeddings = []
        for stream_name in self.streams:
            embeddings.append(self.towers[stream_name](streams[stream_name]))
        z = torch.stack(embeddings, dim=1)                       # (B, S, E)
        use_modality_dropout = False
        if self.training and self.mdrop:
            use_modality_dropout = True
        if apply_mdrop:
            use_modality_dropout = True
        if use_modality_dropout:
            keep = torch.rand(z.shape[0], z.shape[1], device=z.device) > self.p   # (B, S)
            z = z * keep.unsqueeze(-1).float()
        fused, alpha = self.fusion(z)                            # (B, E)
        if self.use_skip and self.skip is not None:
            fused = fused + self.skip(streams)
        return self.head(fused), alpha

# %%
# =============================================================================
# SECTION 7. Metrics and training helpers (gtao_grid.py)
# =============================================================================

def qwk(y_true, y_pred):
    """Quadratic weighted Cohen's kappa between two label vectors.

    Kappa measures agreement beyond chance; "quadratic weights" penalise a
    disagreement by the squared distance between classes, which suits ordinal
    labels. 1 = perfect, 0 = chance level. Returns 0.0 when it is undefined
    (fewer than 2 samples, a single true class, or a non-finite value).
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
    """Convert {name: numpy array} into {name: torch tensor on `device`} (same order)."""
    tensors = {}
    for stream_name, array in streams_dict.items():
        tensors[stream_name] = torch.as_tensor(array, device=device)
    return tensors


@torch.no_grad()
def evaluate(model, val_streams, val_labels):
    """Score the model on a validation set (no modality dropout, no gradients).

    Returns {"qwk": ..., "acc": ..., "mae": ...}.
    """
    logits, _ = model(val_streams, apply_mdrop=False)
    pred = model.head.predict(logits).cpu().numpy()
    y = val_labels.cpu().numpy()
    return {"qwk": qwk(y, pred),
            "acc": round(float((pred == y).mean()), 4),
            "mae": round(float(np.abs(pred - y).mean()), 4)}


def set_seed(seed):
    """Seed PyTorch's and NumPy's global random generators."""
    torch.manual_seed(seed)
    np.random.seed(seed)


def select_rows(streams, row_mask):
    """Return {name: tensor[row_mask]} for every stream (keeps the dict order)."""
    selected = {}
    for stream_name, tensor in streams.items():
        selected[stream_name] = tensor[row_mask]
    return selected


def draw_inner_validation_mask(train_labels, train_subjects):
    """Choose which training-fold rows form the inner validation set.

    Inputs: train_labels (n,) tensor, train_subjects (n,) subject ids.
    A random INNER_FRAC of the training subjects (drawn with np.random, so it
    follows the fold seed) becomes the inner validation set. All rows of a
    subject go to the same side, so inner validation is subject-disjoint from
    inner training. The draw is repeated (up to 20 times) until BOTH sides
    contain at least 2 classes (an inner-validation set with a single class
    would always have QWK 0, and epoch selection would become meaningless).
    Raises RuntimeError if no usable draw is found.
    Returns a numpy bool array (n,), True = inner validation row.
    """
    train_subjects = np.asarray(train_subjects).astype(str)
    unique_subjects = np.array(sorted(set(train_subjects.tolist())))
    n_val = max(1, int(round(INNER_FRAC * len(unique_subjects))))
    found_usable_split = False
    for _ in range(20):                      # re-draw until both sides are usable
        shuffled_positions = np.random.permutation(len(unique_subjects))
        chosen = unique_subjects[shuffled_positions[:n_val]]
        val_subjects = set(chosen.tolist())
        in_val_list = []
        for subject in train_subjects:
            in_val_list.append(subject in val_subjects)
        is_val = np.array(in_val_list)       # (n,) bool, True = inner validation
        all_labels = train_labels.cpu().numpy()
        n_classes_inner_train = len(np.unique(all_labels[~is_val]))
        n_classes_inner_val = len(np.unique(all_labels[is_val]))
        if n_classes_inner_train >= 2 and n_classes_inner_val >= 2:
            found_usable_split = True
            break
    if not found_usable_split:
        raise RuntimeError("could not draw a usable inner split")
    return is_val


def _inner_split(train_streams, train_labels, train_soft, train_subjects):
    """Split the training fold by subject: (1-INNER_FRAC) train, INNER_FRAC val.

    Inputs: train_streams {name: tensor (n, D)}, train_labels (n,) tensor,
    train_soft (n, K) tensor or None, train_subjects (n,) subject ids.
    Uses draw_inner_validation_mask, prints one log line, and returns
    (inner_train_streams, inner_train_labels, inner_train_soft,
     inner_val_streams, inner_val_labels, split_summary)
    where split_summary = {"n_inner_train", "n_inner_val", "inner_val_subjects"}
    is stored in the per-fold JSON record by run_protocol.
    """
    is_val = draw_inner_validation_mask(train_labels, train_subjects)
    subject_array = np.asarray(train_subjects).astype(str)
    inner_val_subjects = sorted(set(subject_array[is_val].tolist()))
    n_inner_train = int((~is_val).sum())
    n_inner_val = int(is_val.sum())
    print(f"    inner split: train n={n_inner_train}, val n={n_inner_val}, "
          f"val subjects={inner_val_subjects}", flush=True)
    split_summary = {"n_inner_train": n_inner_train,
                     "n_inner_val": n_inner_val,
                     "inner_val_subjects": inner_val_subjects}
    val_mask = torch.as_tensor(is_val, device=train_labels.device)
    if train_soft is not None:
        inner_train_soft = train_soft[~val_mask]
    else:
        inner_train_soft = None
    inner_train_streams = select_rows(train_streams, ~val_mask)
    inner_val_streams = select_rows(train_streams, val_mask)
    return (inner_train_streams, train_labels[~val_mask], inner_train_soft,
            inner_val_streams, train_labels[val_mask], split_summary)


def fit_fold(model, train_streams, train_labels, val_streams, val_labels,
             stream_list, loss="coral", train_soft=None, subj_tr=None):
    """Train `model` on one fold with Adam, with early stopping.

    Inputs:
        train_streams {name: (n_tr, D)} tensors, train_labels (n_tr,)
        val_streams   {name: (n_va, D)} tensors, val_labels (n_va,) = outer fold
        stream_list   stream names to feed, loss = "coral" | "hybrid" | "soft"
        train_soft    (n_tr, K) soft labels (only needed for loss="soft")
        subj_tr       (n_tr,) subject id of every training row (needed when NESTED)
    Nested early stopping (GTAO_NESTED=1, default): the training fold is first
    split by subject into inner-train (80%) and inner-validation (20%) with
    _inner_split. The model is trained on inner-train only and the epoch is
    chosen on inner-validation, so the outer fold (val_streams, val_labels
    as passed in) plays no role here; the caller scores it afterwards.
    With GTAO_NESTED=0 (v1 behaviour) training uses the whole training fold
    and the epoch is chosen on the outer fold itself.
    Each epoch: shuffle, mini-batches of BATCH, gradient clipping at 1.0.
    After each epoch the model is evaluated on the (inner or outer)
    validation set and the best weights (highest QWK; ties go to the later
    epoch) are kept. Training stops when there is no improvement for PATIENCE
    epochs (not before MIN_EPOCHS). The best weights are loaded back at the end.
    Returns (best, trace, info):
        best  = {"qwk", "state", "epoch"}  (qwk is the selection-set QWK)
        trace = every 5th epoch: {"epoch", "val_qwk", "gate"} (mean gate weights)
        info  = {"epochs": last epoch run, "loss_last": mean of the last 64 batch losses}
                plus, only when NESTED, "inner_split": {"n_inner_train",
                "n_inner_val", "inner_val_subjects"}
    """
    inner_split_summary = None
    if NESTED:
        if subj_tr is None:
            raise ValueError("NESTED=1 requires subj_tr (training-fold subject ids)")
        inner = _inner_split(train_streams, train_labels, train_soft, subj_tr)
        train_streams = inner[0]
        train_labels = inner[1]
        train_soft = inner[2]
        val_streams = inner[3]       # from here on "validation" = inner validation
        val_labels = inner[4]
        inner_split_summary = inner[5]
    # Draw a new seed from NumPy's generator (seeded per fold by the caller).
    set_seed(int(np.random.randint(2 ** 31)))
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    best = {"qwk": -1.0, "state": None, "epoch": 0}
    trace = []
    losses = []
    for epoch in range(1, EPOCHS + 1):
        model.train()
        order = torch.randperm(len(train_labels))
        for start in range(0, len(train_labels), BATCH):
            batch_idx = order[start:start + BATCH]
            optimizer.zero_grad()
            batch_streams = {}
            for stream_name in stream_list:
                batch_streams[stream_name] = train_streams[stream_name][batch_idx]
            logits, _ = model(batch_streams)
            batch_labels = train_labels[batch_idx]
            if train_soft is not None:
                batch_soft = train_soft[batch_idx]
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
            saved_state = {}
            for name, value in model.state_dict().items():
                saved_state[name] = value.detach().clone()
            best = {"qwk": val_scores["qwk"],
                    "state": saved_state,
                    "epoch": epoch}
        if epoch % 5 == 0:
            with torch.no_grad():
                val_inputs = {}
                for stream_name in stream_list:
                    val_inputs[stream_name] = val_streams[stream_name]
                _, alpha = model(val_inputs, apply_mdrop=False)
            gate = []
            if alpha is not None:
                for weight in alpha.mean(dim=0).cpu().numpy():
                    gate.append(round(float(weight), 4))
            trace.append({"epoch": epoch, "val_qwk": val_scores["qwk"], "gate": gate})
        if epoch >= MIN_EPOCHS and epoch - best["epoch"] >= PATIENCE:
            break
    model.load_state_dict(best["state"])
    model.eval()
    info = {"epochs": epoch, "loss_last": round(float(np.mean(losses[-64:])), 4)}
    if inner_split_summary is not None:
        info["inner_split"] = inner_split_summary
    return best, trace, info


def cfg_name(cfg):
    """Folder / table name of a config, e.g. "gate_skip1_drop0_soft".

    The "_coral" suffix is omitted for the default loss. Unimodal configs are
    "unimodal_<stream>", the shuffle control is "shuffle_core_skip1_drop1".
    """
    if cfg.get("unimodal"):
        return f"unimodal_{cfg['unimodal']}"
    if cfg.get("shuffle"):
        return "shuffle_core_skip1_drop1"
    base = f"{cfg['fusion']}_skip{int(cfg['use_skip'])}_drop{int(cfg['mdrop'])}"
    if cfg.get("loss", "coral") == "coral":
        return base
    return base + f"_{cfg['loss']}"

# %%
# =============================================================================
# SECTION 8. Grid drivers: train one config over a list of folds
# =============================================================================
# run_protocol(corpus, folds, cfg, seed) trains `cfg` on each (train, val) fold
# for one seed and pools the out-of-fold (OOF) predictions.
# P2: 5 subject-grouped folds; P1: a single subject-filtered train/val pair.

def corpus_streams(corpus, stream_list, rows):
    """Return {stream: torch tensor of corpus["feat_<stream>"][rows]} in stream_list order."""
    arrays = {}
    for s in stream_list:
        arrays[s] = corpus[f"feat_{s}"][rows]
    return to_torch(arrays)


def pooled_scores(y_true, y_pred_float, ok):
    """QWK / acc / MAE of the rows where `ok` is True (predictions stored as floats)."""
    true_int = y_true[ok].astype(int)
    pred_int = y_pred_float[ok].astype(int)
    return {"qwk": round(qwk(true_int, pred_int), 4),
            "acc": round(float(np.mean(pred_int == true_int)), 4),
            "mae": round(float(np.mean(np.abs(pred_int - true_int))), 4),
            "n": int(ok.sum())}


def run_protocol(corpus, folds, cfg, seed):
    """Train one config on every fold for one seed; return a JSON-ready dict.

    Returns meta = {"probe": None, "folds": [...], "gate_trace": {fold: trace},
    "oof": {...per clip...}, "pooled": {"qwk", "acc", "mae", "n", "n_subjects"}}
    or None if no usable stream exists.
    For the shuffle control the labels are permuted once over the whole
    population before training (rng seeded with `seed`).
    """
    ys = corpus["y"]
    soft_full = corpus["soft"]
    if cfg.get("unimodal"):
        stream_list = [cfg["unimodal"]]
    else:
        stream_list = []
        for s in STREAMS:
            if corpus[f"feat_{s}"].shape[0] > 0:
                stream_list.append(s)
    dims = {}
    for s in stream_list:
        if corpus[f"feat_{s}"].shape[0] > 0:
            dims[s] = int(corpus[f"feat_{s}"].shape[1])
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

    oof_pred = np.full(len(ys), np.nan)   # NaN = clip not predicted (yet)
    meta = {"probe": None, "folds": [], "gate_trace": {}}
    for f, (tr, te) in enumerate(folds):
        # NOTE: te.any() is True when some index in te is non-zero (it is not
        # an emptiness test); kept exactly as in the original.
        if len(set(y_run[tr])) < 2 or not te.any():
            continue
        train_streams = corpus_streams(corpus, stream_list, tr)
        train_labels = torch.as_tensor(y_run[tr], device=device)
        if cfg.get("loss") == "soft":
            train_soft = torch.as_tensor(soft_run[tr], device=device)
        else:
            train_soft = None
        val_streams = corpus_streams(corpus, stream_list, te)
        val_labels = torch.as_tensor(y_run[te], device=device)
        set_seed(int(f"{seed}{f}"))       # e.g. seed 1, fold 3 -> 13
        if cfg.get("unimodal"):
            model = GTAONetGrid(dims, fusion="core", use_skip=False,
                                mdrop=False).to(device)
        else:
            model = GTAONetGrid(dims, fusion=cfg["fusion"], use_skip=cfg["use_skip"],
                                mdrop=cfg["mdrop"]).to(device)
        t0 = time.time()
        best, trace, info = fit_fold(model, train_streams, train_labels,
                                     val_streams, val_labels,
                                     subj_tr=corpus["subjects"][tr],
                                     stream_list=stream_list,
                                     loss=cfg.get("loss", "coral"),
                                     train_soft=train_soft)
        with torch.no_grad():
            logits, _ = model(val_streams, apply_mdrop=False)
            oof_pred[te] = model.head.predict(logits).cpu().numpy()
        meta["gate_trace"][f] = trace
        fold_entry = {"fold": f, "n_train": int(len(tr)),
                      "n_val": int(len(te)), "best_val_qwk": best["qwk"],
                      "best_epoch": best["epoch"]}
        for info_key, info_value in info.items():
            if info_key == "inner_split":
                continue                  # added after wall_s, see below
            fold_entry[info_key] = info_value
        fold_entry["wall_s"] = round(time.time() - t0, 1)
        if "inner_split" in info:
            # Nested mode only: record which inner split chose the epoch.
            fold_entry["n_inner_train"] = info["inner_split"]["n_inner_train"]
            fold_entry["n_inner_val"] = info["inner_split"]["n_inner_val"]
            fold_entry["inner_val_subjects"] = info["inner_split"]["inner_val_subjects"]
        meta["folds"].append(fold_entry)
        print(f"  fold {f}: val n={len(te)} QWK={best['qwk']:.4f} "
              f"ep={best['epoch']}  [{round(time.time() - t0, 1)}s]", flush=True)

    ok = np.isfinite(oof_pred)
    meta["oof"] = {"keys": corpus["keys"][ok].astype(str).tolist(),
                   "y": ys[ok].astype(int).tolist(),
                   "pred": oof_pred[ok].astype(int).tolist(),
                   "subject": corpus["subjects"][ok].astype(str).tolist(),
                   "split": corpus["splits"][ok].astype(str).tolist()}
    pooled = pooled_scores(ys, oof_pred, ok)
    pooled["n_subjects"] = len(set(corpus["subjects"][ok].tolist()))
    meta["pooled"] = pooled
    return meta


def assign_subject_folds(subject_ids, seed, n_folds):
    """Give every subject a fold number 0..n_folds-1 (random but seeded).

    Unique subjects are sorted, shuffled with np.random.default_rng(seed),
    then dealt out round-robin. Returns an array with the fold of each entry
    of subject_ids.
    """
    unique_subjects = sorted(set(subject_ids.tolist()))
    rng = np.random.default_rng(seed)
    rng.shuffle(unique_subjects)
    fold_of_subject = {}
    for i, subject in enumerate(unique_subjects):
        fold_of_subject[subject] = i % n_folds
    fold_list = []
    for subject in subject_ids:
        fold_list.append(fold_of_subject[subject])
    return np.array(fold_list)


def subject_group_folds(corpus, seed, n_folds=N_FOLDS):
    """P2 folds: list of (train indices, val indices); each subject is in one val fold."""
    fold_of = assign_subject_folds(corpus["subjects"], seed, n_folds)
    folds = []
    for f in range(n_folds):
        folds.append((np.flatnonzero(fold_of != f), np.flatnonzero(fold_of == f)))
    return folds


def official_split_folds(corpus):
    """P1: train on Train minus subjects seen in Validation; eval on Validation.

    Returns a list with one (train indices, val indices) pair.

    History: tools/make_nested.py once inserted the nested-early-stopping
    block here by mistake (it belongs in fit_fold, where it now is). With
    GTAO_NESTED=1 that misplaced block crashed with NameError at the start of
    P1; it has been removed from this function.
    """
    splits = corpus["splits"]
    train_mask = splits == "Train"
    val_mask = splits == "Validation"
    # Subjects that appear in Validation are removed from training.
    leak_subjects = set(corpus["subjects"][val_mask].tolist())
    not_leaked_list = []
    for subject in corpus["subjects"]:
        not_leaked_list.append(subject not in leak_subjects)
    clean_train_mask = train_mask & np.array(not_leaked_list)
    return [(np.flatnonzero(clean_train_mask), np.flatnonzero(val_mask))]


def summary_sort_key(row):
    """Sort summary rows by QWK (best first), then alphabetically by config name."""
    return (-row["qwk_mean"], row["config"])


def summarize(configs, doc_key, run_dir, write_to):
    """Aggregate the per-seed oof.json files of each config into summary rows.

    For each config: mean / std of pooled QWK over seeds, mean acc and MAE,
    median n, and the final gate weights of seed 0 / first fold. Prints a
    table, checks the shuffle control, and stores the rows in the JSON file
    `write_to` under doc_key -> "rows". Returns the rows.
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
            run_meta = json.loads(oof_file.read_text())
            if not run_meta.get("pooled"):
                continue
            pooled = run_meta["pooled"]
            qwks.append(pooled["qwk"])
            accs.append(pooled["acc"])
            maes.append(pooled["mae"])
            ns.append(pooled["n"])
            probes.append(run_meta.get("probe"))
            gates.append(run_meta.get("gate_trace"))
        if not qwks:
            continue
        if cfg.get("unimodal"):
            fusion_label = "unimodal"
        else:
            fusion_label = cfg["fusion"]
        qwk_per_seed = []
        for q in qwks:
            qwk_per_seed.append(round(q, 4))
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
        if gates and gates[0]:
            # gate trace of the first fold (fold keys are strings after JSON)
            first_fold_trace = sorted(gates[0].items())[0][1]
            if first_fold_trace:
                rows[-1]["gate_alpha_final"] = first_fold_trace[-1]["gate"]
    rows.sort(key=summary_sort_key)
    if "P2" in doc_key:
        pooling_text = "over 5 folds"
    else:
        pooling_text = "train/val"
    print(f"\n=== {doc_key} (pooled {pooling_text}"
          f", seeds {list(SEEDS)}) ===")
    print(f"{'config':28s} {'QWK mean±std':>14s} {'acc':>7s} {'mae':>6s} {'n':>5s}")
    for row in rows:
        print(f"{row['config']:28s} {row['qwk_mean']:+.4f}±{row['qwk_std']:.3f} "
              f"{row['acc_mean']:7.3f} {row['mae_mean']:6.3f} {row['n_median']:5d}")

    # Shuffle control: with permuted labels the QWK must stay near 0.
    shuffle_rows = []
    for row in rows:
        if row.get("shuffle"):
            shuffle_rows.append(row)
    if shuffle_rows:
        max_abs_qwk = None
        for row in shuffle_rows:
            value = abs(row["qwk_mean"])
            if max_abs_qwk is None or value > max_abs_qwk:
                max_abs_qwk = value
        if max_abs_qwk <= SHUFFLE_TOL:
            verdict = "PASS"
        else:
            verdict = "FAIL"
        print(f"shuffle control: max |QWK| = {max_abs_qwk:.4f} "
              f"({verdict} ≤ {SHUFFLE_TOL})")
    unimodal_rows = []
    for row in rows:
        if row.get("unimodal"):
            unimodal_rows.append(row)
    if unimodal_rows:
        best_row = None
        for row in unimodal_rows:
            if best_row is None or row["qwk_mean"] > best_row["qwk_mean"]:
                best_row = row
        print(f"best in-protocol unimodal: {best_row['config']} = {best_row['qwk_mean']:.4f}")

    if write_to.exists():
        doc = json.loads(write_to.read_text())
    else:
        doc = {}
    doc.setdefault(doc_key, {})["rows"] = rows
    write_to.write_text(json.dumps(doc, indent=2))
    return rows


def masked_eval(corpus, seeds=SEEDS):
    """H4: retrain gate/core skip1-drop0-soft; zero out each stream at eval.

    Uses only clips that have every stream. For each seed, 5 subject-grouped
    folds are built (same recipe as P2). After training, predictions are made
    with all streams ("none") and with each stream replaced by zeros
    ("mask_<stream>"). Returns {seed: {config name: {"config", "seed",
    "conditions": {condition: {"qwk", "acc", "mae", "n", ["gate_alpha"]}}}}}.
    """
    configs = ({'fusion': 'gate', 'use_skip': True, 'mdrop': False, 'loss': 'soft'},
               {'fusion': 'core', 'use_skip': True, 'mdrop': False, 'loss': 'soft'})
    all_present = []
    for s in STREAMS:
        if corpus[f"feat_{s}"].shape[0] > 0:
            all_present.append(s)
    keep = np.ones(len(corpus["y"]), dtype=bool)
    for s in all_present:
        keep &= corpus[f"avail_{s}"]
    idx_all = np.flatnonzero(keep)
    ys = corpus["y"]
    per_seed = {}
    for seed in seeds:
        fold_of = assign_subject_folds(corpus["subjects"][idx_all], seed, N_FOLDS)
        rows = {}
        for cfg in configs:
            name = cfg_name(cfg)
            condition_names = ["none"]
            for s in all_present:
                condition_names.append(f"mask_{s}")
            cond_pred = {}
            for condition in condition_names:
                cond_pred[condition] = np.full(len(corpus["y"]), np.nan)
            alpha_acc = {}
            for condition in cond_pred:
                alpha_acc[condition] = []
            dims = {}
            for s in all_present:
                dims[s] = int(corpus[f"feat_{s}"].shape[1])
            for f in range(N_FOLDS):
                te = idx_all[fold_of == f]
                tr = idx_all[~np.isin(idx_all, te)]
                if len(set(ys[tr])) < 2 or len(te) == 0:
                    continue
                train_streams = corpus_streams(corpus, all_present, tr)
                train_labels = torch.as_tensor(ys[tr], device=device)
                train_soft = torch.as_tensor(corpus["soft"][tr], device=device)
                val_streams = corpus_streams(corpus, all_present, te)
                val_labels = torch.as_tensor(ys[te], device=device)
                set_seed(int(f"{seed}{f}"))
                model = GTAONetGrid(dims, fusion=cfg["fusion"],
                                    use_skip=cfg["use_skip"], mdrop=cfg["mdrop"])
                model = model.to(device)
                best, _, _ = fit_fold(model, train_streams, train_labels,
                                      val_streams, val_labels,
                                      subj_tr=corpus["subjects"][tr],
                                      stream_list=all_present,
                                      loss=cfg["loss"], train_soft=train_soft)
                # (fit_fold already loaded the best weights; reloading is harmless.)
                model.load_state_dict(best["state"])
                model.eval()
                with torch.no_grad():
                    logits, alpha = model(val_streams, apply_mdrop=False)
                    cond_pred["none"][te] = model.head.predict(logits).cpu().numpy()
                    if alpha is not None:
                        alpha_acc["none"].append(alpha.cpu().numpy())
                    for s in all_present:
                        # Same inputs, except stream s is replaced by zeros.
                        masked_streams = {}
                        for stream_name, value in val_streams.items():
                            if stream_name == s:
                                masked_streams[stream_name] = torch.zeros_like(value)
                            else:
                                masked_streams[stream_name] = value
                        logits_m, alpha_m = model(masked_streams, apply_mdrop=False)
                        cond_pred[f"mask_{s}"][te] = (
                            model.head.predict(logits_m).cpu().numpy())
                        if alpha_m is not None:
                            alpha_acc[f"mask_{s}"].append(alpha_m.cpu().numpy())
            out = {"config": cfg, "seed": seed, "conditions": {}}
            for condition, pred in cond_pred.items():
                ok = np.isfinite(pred)
                entry = pooled_scores(ys, pred, ok)
                if alpha_acc[condition]:
                    all_alpha = np.concatenate(alpha_acc[condition], axis=0)   # (n, S)
                    gate_alpha = []
                    for x in all_alpha.mean(axis=0):
                        gate_alpha.append(round(float(x), 4))
                    entry["gate_alpha"] = gate_alpha
                out["conditions"][condition] = entry
            rows[name] = out
            # Largest QWK drop caused by masking one stream.
            max_drop = None
            for condition, values in out['conditions'].items():
                if condition.startswith('mask_'):
                    drop = out['conditions']['none']['qwk'] - values['qwk']
                    if max_drop is None or drop > max_drop:
                        max_drop = drop
            print(f"  H4 {name} seed {seed}: "
                  f"none {out['conditions']['none']['qwk']:.3f} -> "
                  f"max masked drop "
                  f"{max_drop:.3f}")
        per_seed[seed] = rows
    return per_seed

# %%
# =============================================================================
# SECTION 9. Protocol P1 (official split, subject-filtered): direct SOTA comparison
# =============================================================================

def build_fusion_configs():
    """All 24 fusion configs: FUSIONS x SKIPS x MDROPS x LOSSES (in that order)."""
    configs = []
    for fusion, use_skip, mdrop, loss in product(FUSIONS, SKIPS, MDROPS, LOSSES):
        configs.append({"fusion": fusion, "use_skip": use_skip, "mdrop": mdrop,
                        "loss": loss})
    return configs


def run_p1(corpus):
    """Train every config x seed on the official split and write the summary.

    Already finished runs (existing oof.json) are skipped, so a crashed
    Kaggle session can be resumed. The training subjects passed on to
    fit_fold (for the nested inner split) are those of the cleaned Train set.
    Returns the summary rows (used by run_p1_test to pick its two configs).
    """
    run_dir = RUNS / "P1_official"
    folds = official_split_folds(corpus)
    tr, te = folds[0]
    print(f"P1 | train n={len(tr)}  val n={len(te)}  "
          f"train subjects={len(set(corpus['subjects'][tr].tolist()))}  "
          f"val subjects={len(set(corpus['subjects'][te].tolist()))}")
    configs = build_fusion_configs()
    for s in STREAMS:
        configs.append({"unimodal": s, "loss": "coral"})
    for cfg in configs:
        for seed in SEEDS:
            oof_file = run_dir / cfg_name(cfg) / f"seed{seed}" / "oof.json"
            if oof_file.exists():
                continue
            run_meta = run_protocol(corpus, folds, cfg, seed)
            if run_meta is None:
                continue
            oof_file.parent.mkdir(parents=True, exist_ok=True)
            oof_file.write_text(json.dumps(run_meta, indent=1))
    rows = summarize(configs, "protocol_P1_official", run_dir, METRICS)
    return rows

# %%
# =============================================================================
# SECTION 9b. Protocol P1-Test (official Test partition): comparison with papers
# =============================================================================
# Published DAiSEE results are reported on the official Test partition, usually
# with Train and Validation merged for training. P1 above scores Validation, so
# its numbers cannot be set beside theirs directly.
#
# Choosing which configuration to report by its Test score would be selection on
# the scored set (the mistake nested early stopping removes). So the two models
# scored here are chosen BEFORE Test is touched: the best single stream and the
# best fusion model by their P1 (Validation) QWK. Each is then retrained on
# Train + Validation, with the usual nested inner split for epoch selection, and
# predicts Test once.

def test_split_folds(corpus):
    """P1-Test: train on Train + Validation, score on Test.

    Subjects that also appear in Test are removed from training, as P1 does
    for Validation. Returns a list with one (train indices, test indices)
    pair, or an empty list when the corpus has no Test clips.
    """
    splits = corpus["splits"]
    train_mask = (splits == "Train") | (splits == "Validation")
    test_mask = splits == "Test"
    if not test_mask.any():
        return []
    leak_subjects = set(corpus["subjects"][test_mask].tolist())
    not_leaked_list = []
    for subject in corpus["subjects"]:
        not_leaked_list.append(subject not in leak_subjects)
    clean_train_mask = train_mask & np.array(not_leaked_list)
    return [(np.flatnonzero(clean_train_mask), np.flatnonzero(test_mask))]


def best_row(rows, want_unimodal):
    """Highest-QWK summary row among single streams (True) or fusion models (False)."""
    best = None
    for row in rows:
        if row.get("shuffle"):
            continue
        is_unimodal = row["fusion"] == "unimodal"
        if is_unimodal != want_unimodal:
            continue
        if best is None or row["qwk_mean"] > best["qwk_mean"]:
            best = row
    return best


def run_p1_test(corpus, p1_rows):
    """Retrain the P1-selected best single stream and best fusion model on
    Train + Validation and score them on the official Test partition.

    Writes daisee_gt_grid/P1_test/<config>/seed<k>/oof.json and the
    "protocol_P1_test" rows of METRICS (each row also records that it was
    selected on Validation).
    """
    folds = test_split_folds(corpus)
    if not folds:
        print("P1-Test skipped: the corpus has no Test clips")
        return []
    tr, te = folds[0]
    print(f"P1-Test | train n={len(tr)}  test n={len(te)}  "
          f"train subjects={len(set(corpus['subjects'][tr].tolist()))}  "
          f"test subjects={len(set(corpus['subjects'][te].tolist()))}")

    # Map config names back to config dicts, then take the two P1 winners.
    all_configs = build_fusion_configs()
    for s in STREAMS:
        all_configs.append({"unimodal": s, "loss": "coral"})
    config_by_name = {}
    for cfg in all_configs:
        config_by_name[cfg_name(cfg)] = cfg
    chosen = []
    for want_unimodal in (True, False):
        row = best_row(p1_rows, want_unimodal)
        if row is not None:
            chosen.append(config_by_name[row["config"]])
            print(f"P1-Test: scoring {row['config']} "
                  f"(chosen on Validation, QWK {row['qwk_mean']:.4f})")

    run_dir = RUNS / "P1_test"
    for cfg in chosen:
        for seed in SEEDS:
            oof_file = run_dir / cfg_name(cfg) / f"seed{seed}" / "oof.json"
            if oof_file.exists():
                continue
            run_meta = run_protocol(corpus, folds, cfg, seed)
            if run_meta is None:
                continue
            oof_file.parent.mkdir(parents=True, exist_ok=True)
            oof_file.write_text(json.dumps(run_meta, indent=1))
    rows = summarize(chosen, "protocol_P1_test", run_dir, METRICS)
    doc = json.loads(METRICS.read_text())
    doc["protocol_P1_test"]["selected_on"] = "protocol_P1_official (Validation QWK)"
    METRICS.write_text(json.dumps(doc, indent=2))
    return rows

# %%
# =============================================================================
# SECTION 10. Protocol P2 (GroupKFold-5 by subject, full factorial): DIPSER-style
# =============================================================================

def run_p2(corpus):
    """Train every config x seed on 5 subject-grouped folds, then run H4.

    Configs: 24 fusion configs, 3 unimodal baselines and the shuffle control.
    Writes per-run oof.json files, the P2 summary rows and the
    "h4_masked_eval_P2" block of METRICS. Returns the summary rows.
    """
    run_dir = RUNS / "P2_groupkfold"
    configs = build_fusion_configs()
    for s in STREAMS:
        configs.append({"unimodal": s, "loss": "coral"})
    configs.append({"shuffle": True, "fusion": "core", "use_skip": True,
                    "mdrop": True, "loss": "coral"})
    for seed in SEEDS:
        folds = subject_group_folds(corpus, seed)
        class_counts = np.bincount(corpus["y"])
        print(f"P2 seed {seed} | n={len(corpus['y'])} "
              f"subjects={len(set(corpus['subjects'].tolist()))} "
              f"class_dist={class_counts.tolist()}")
        for cfg in configs:
            name = cfg_name(cfg)
            oof_file = run_dir / name / f"seed{seed}" / "oof.json"
            if oof_file.exists():
                continue
            run_meta = run_protocol(corpus, folds, cfg, seed)
            if run_meta is None:
                continue
            oof_file.parent.mkdir(parents=True, exist_ok=True)
            oof_file.write_text(json.dumps(run_meta, indent=1))
    rows = summarize(configs, "protocol_P2_groupkfold", run_dir, METRICS)
    h4_key = "h4_masked_eval_P2"
    if METRICS.exists():
        doc = json.loads(METRICS.read_text())
    else:
        doc = {}
    # A resumed run (copy_previous_outputs) already has the H4 results: keep them.
    if len(doc.get(h4_key, {}).get("per_seed", {})) == len(SEEDS):
        print("H4 masked evaluation already done in the earlier run: kept")
        return rows
    h4 = masked_eval(corpus, seeds=SEEDS)
    # Keep only the "conditions" part of each H4 result:
    # {"0": {config name: {condition: scores}}, "1": ..., "2": ...}
    per_seed_doc = {}
    for seed, results_by_config in h4.items():
        seed_doc = {}
        for config_name, out in results_by_config.items():
            conditions = {}
            for condition in out["conditions"]:
                conditions[condition] = out["conditions"][condition]
            seed_doc[config_name] = conditions
        per_seed_doc[str(seed)] = seed_doc
    doc.setdefault(h4_key, {})["per_seed"] = per_seed_doc
    METRICS.write_text(json.dumps(doc, indent=2))
    return rows

# %%
# =============================================================================
# SECTION 11. Main
# =============================================================================

def analyze(corpus):
    """Print clip / subject / class counts per split and subject overlap between splits."""
    splits = corpus["splits"]
    subjects = corpus["subjects"]
    print("\n== population ==")
    for split_name in ("Train", "Validation", "Test"):
        mask = splits == split_name
        print(f"  {split_name:10s} clips={int(mask.sum()):5d} "
              f"subjects={len(set(subjects[mask].tolist())):3d} "
              f"class={np.bincount(corpus['y'][mask], minlength=N_CLASSES).tolist()}")
    overlap = {}
    for a, b in [("Train", "Validation"), ("Train", "Test"), ("Validation", "Test")]:
        subjects_a = set(subjects[splits == a].tolist())
        subjects_b = set(subjects[splits == b].tolist())
        overlap[a + "|" + b] = len(subjects_a & subjects_b)
    print(f"  subject overlap (Kaggle splits, NOT subject-disjoint): {overlap}")


def keep_complete_clips(corpus):
    """Keep only the clips in which all three streams were extracted.

    load_daisee() fills a missing face-mesh or head-pose stream with NaN. A single NaN
    row inside a training batch would turn the model weights into NaN (this happened
    on EngageNet in September 2026). The paper describes the DAiSEE population as the
    clips with all three streams, so we enforce that here.

    In the v1 run every clip had all three streams (the masked evaluation, which
    applies the same filter, kept all 7,919 clips). In that case the corpus is
    returned unchanged, so results stay identical to v1.
    """
    complete = np.ones(len(corpus["y"]), dtype=bool)
    for stream in STREAMS:
        complete = complete & corpus[f"avail_{stream}"]
    n_total = len(complete)
    n_complete = int(complete.sum())
    if n_complete == n_total:
        print(f"all {n_total} clips have every stream")
        return corpus

    print(f"WARNING: {n_total - n_complete} of {n_total} clips miss a stream; "
          f"keeping the {n_complete} complete clips")
    # A cached corpus is an NpzFile (names in .files); a freshly built one is a dict.
    if hasattr(corpus, "files"):
        names = corpus.files
    else:
        names = list(corpus.keys())
    filtered = {}
    for name in names:
        array = corpus[name]
        if array.ndim >= 1 and array.shape[0] == n_total:
            filtered[name] = array[complete]
        else:
            filtered[name] = array      # e.g. an empty (0, 0) stream placeholder
    return filtered


def find_previous_output():
    """Folder holding the outputs of an earlier run of this script, or None.

    On Kaggle: an earlier version of this kernel attached as an input (its
    output is mounted somewhere under /kaggle/input). Locally: the folder in
    the environment variable GTAO_RESUME_FROM. Only a few folder levels are
    searched, so the large DAiSEE video tree is not walked.
    """
    resume_from = os.environ.get("GTAO_RESUME_FROM", "")
    if resume_from:
        return Path(resume_from)
    if not ON_KAGGLE:
        return None
    for pattern in ("*/daisee_gt_metrics.json", "*/*/daisee_gt_metrics.json",
                    "*/*/*/daisee_gt_metrics.json"):
        for metrics_file in Path("/kaggle/input").glob(pattern):
            return metrics_file.parent
    return None


def copy_previous_outputs():
    """Copy an earlier run's cache, metrics and oof.json files into OUT.

    Finished runs are then skipped (each protocol skips a config and seed whose
    oof.json exists) and the clip features are loaded from the cache instead of
    decoding every video again. Files already in OUT are never overwritten.
    With no earlier run attached, nothing happens and the run starts fresh.
    """
    previous = find_previous_output()
    if previous is None:
        return
    n_copied = 0
    for source in previous.rglob("*"):
        if not source.is_file():
            continue
        relative = source.relative_to(previous)
        is_cache = relative.name == CACHE.name
        is_metrics = relative.name == METRICS.name
        is_run = relative.parts[0] == RUNS.name
        if not (is_cache or is_metrics or is_run):
            continue
        target = OUT / relative
        if target.exists():
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        n_copied += 1
    print(f"reused {n_copied} files from the earlier run in {previous}")


def main():
    """Load (or build) the corpus, then run P1, P2 + H4 and P1-Test, updating METRICS."""
    t0 = time.time()
    copy_previous_outputs()
    corpus = load_daisee()
    corpus = keep_complete_clips(corpus)
    METRICS.parent.mkdir(parents=True, exist_ok=True)
    if METRICS.exists():
        doc = json.loads(METRICS.read_text())
    else:
        doc = {}
    doc["status"] = "running"
    doc["env"] = {"device": device, "fast": FAST, "host": socket.gethostname()}
    METRICS.write_text(json.dumps(doc, indent=2))
    analyze(corpus)
    p1_rows = run_p1(corpus)
    run_p2(corpus)
    # Last, so that P1 and P2 consume the random generator exactly as before.
    run_p1_test(corpus, p1_rows)
    doc = json.loads(METRICS.read_text())
    doc["status"] = "done"
    doc["finished_utc"] = time.strftime("%Y-%m-%d %H:%M:%S")
    doc["wall_s"] = round(time.time() - t0, 1)
    METRICS.write_text(json.dumps(doc, indent=2))
    print("total wall:", round(time.time() - t0, 1), "s")


if __name__ == "__main__":
    main()
