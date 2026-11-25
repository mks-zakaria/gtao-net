"""gTAO-Net attention grid on DIPSER (v3: v2 nested early stopping + input
standardisation + person-grouped stratified folds; see GTAO_VARIANTS below).

What this script does
=====================
It trains and evaluates many small neural networks that predict how attentive a
student is in a classroom (3 classes: low / medium / high attention) from four
"streams" (modalities) of sensor data recorded during class:

* ``face``       - grey-scale face frames from a camera,
* ``facemesh``   - 478 three-dimensional face landmarks per frame,
* ``head_pose``  - pitch / yaw / roll of the head per frame,
* ``physio``     - smart-watch signals (heart rate, accelerometer, gyroscope, light).

Each recording is cut into 4-second windows. Every window gets one label: the
plurality vote of several human raters (their 1..5 scores collapsed to 3 classes).
Every stream in a window is compressed with Tchebichef moments (a set of discrete
orthogonal polynomials, similar in spirit to a cosine transform) into a short
fixed-length feature vector.

The experiment grid
===================
The main model is gTAO-Net: one small network ("tower") per stream, a fusion
block that combines the towers, and an ordinal (CORAL) output head. The grid
tries every combination of four design choices::

    fusion            {gate, core}        gate can switch a stream off, core cannot
    global skip       {on, off}           extra linear path from inputs to the head
    modality dropout  {on, off}           randomly hide whole streams while training
    loss              {coral, hybrid, soft}

That is 2 x 2 x 2 x 3 = 24 fusion configurations. In addition it trains
4 unimodal CORAL baselines (one stream each, ``unimodal_<stream>_cw``),
1 "shuffle control" (labels randomly permuted, so the score must stay close
to 0) and 4 unimodal soft-target baselines (``unimodal_<stream>_cw_soft``):
24 + 4 + 1 + 4 = 33 configurations. Every configuration is evaluated with
5-fold cross-validation where the folds are split BY SUBJECT (a student is
never in both train and test), repeated for 3 random seeds (0, 1, 2):
33 x 5 x 3 = 495 model fits (plus 30 for the H4 probe below).
Afterwards an extra probe ("H4 masked eval") retrains two configurations and
measures how much the score drops when one stream is zeroed at test time.

The score is QWK (quadratic weighted Cohen's kappa): 1 = perfect agreement with
the raters, 0 = chance level, negative = worse than chance.

Nested early stopping (what is new in v2)
=========================================
Training runs for at most EPOCHS epochs and keeps the epoch with the best
validation QWK ("early stopping"). In v1 that validation set was the very test
fold that was then reported, which makes the reported numbers slightly too
optimistic. In v2 (the default, ``GTAO_NESTED=1``) about 20 % of the TRAINING
subjects of each fold are held back as an "inner" validation set used only to
choose the epoch; the test fold is touched only once, for the final prediction.
Both inner parts must contain at least 2 classes (the split is redrawn up to
20 times). Each fold record in ``oof.json`` then also stores ``n_inner_train``,
``n_inner_val`` and ``inner_val_subjects``.
Setting ``GTAO_NESTED=0`` reproduces v1 exactly (for the configurations v1 has).

Data it expects
===============
A folder of per-subject bundles named
``<group>__<experiment>__<subject>.zipbin`` (ZIP archives). Inside each bundle
there is a subject folder containing ``images/*.png`` (file name = time stamp),
``metadata/<same name>.json`` (facemesh and head pose), ``labels/*.json`` (one
file per rater with attention change points) and ``watch_sensors/*.json``.
On Kaggle this is the dataset ``dipseer-paper1-slice`` mounted under
``/kaggle/input``. Locally, set ``DIPSEER_DATA_ROOT`` to such a folder
(``tools/make_synthetic_dipser.py`` writes a small fake one for testing).

How to run
==========
* Kaggle: upload as a "script" kernel with the dataset attached and run it.
  Results are written to ``/kaggle/working``.
* Locally (quick CPU smoke test, seed 0 only, 8 epochs, at most 6 subjects)::

      DIPSEER_DATA_ROOT=/path/to/zipbins GRID_FAST=1 python gtao_grid.py

  Add ``GTAO_NESTED=0`` to get the original v1 behaviour.
  Results are written to ``/tmp/gtao_grid_out``.

Files it writes (inside the output folder)
==========================================
* ``corpus_grid_v1.npz``       - encoded feature cache (re-used if present).
* ``gtao_grid_v1/<config>/seed<k>/oof.json`` - per-fold details and the
  out-of-fold predictions of one configuration and seed.
* ``grid_v1_metrics.json``     - corpus statistics plus the summary table.
* ``h4_masked_eval.json``      - results of the masked-stream probe.
"""
import json
import math
import os
import shutil
import tempfile
import time
import zipfile
from pathlib import Path

import mpmath
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import cohen_kappa_score


# ---------------------------------------------------------------------------
# 1. Settings
# ---------------------------------------------------------------------------
ON_KAGGLE = os.path.isdir("/kaggle/working")
if ON_KAGGLE:
    OUT = Path("/kaggle/working")
else:
    OUT = Path("/tmp/gtao_grid_out")
INPUT_DIR = Path("/kaggle/input/dipseer-paper1-slice")
RUNS = OUT / "gtao_grid_v1"                       # one sub-folder per config/seed
RUNS.mkdir(parents=True, exist_ok=True)
METRICS = OUT / "grid_v1_metrics.json"

FAST = os.environ.get("GRID_FAST", "") == "1"   # local synthetic smoke test
WINDOW_S = 4.0                                  # window length in seconds
FACE_SIZE = 64                                  # face frames are resized to 64x64
FACE_K = 16                                     # 2-D Tchebichef order -> 16x16 per frame
SEQ_K = 6                                       # 1-D Tchebichef order across time
N_CLASSES = 3
EMB = 64                                        # size of each tower's output vector
HID = 128                                       # hidden size inside a tower
LR = 3e-3
if FAST:
    EPOCHS = 8
else:
    EPOCHS = 40
BATCH = 32
PATIENCE = 8                                    # stop after 8 epochs without improvement
# v2: nested early stopping. Epoch selection uses an inner validation split of
# the TRAINING-fold subjects; the scored outer fold never influences training.
NESTED = os.environ.get("GTAO_NESTED", "1") == "1"
INNER_FRAC = 0.2                                # share of training subjects held back
MIN_EPOCHS = 5
N_FOLDS = 5
SEEDS = (0, 1, 2)
MDROP_P = 0.2                                   # modality dropout probability

# v3: input standardisation and fold construction (Phase 1 of PLAN_accuracy.md).
# GTAO_VARIANTS lists the (norm:folds) variants run one after the other, each in
# its own output folder. Defaults are pre-declared before any v3 result exists:
#   norm  none    : raw Tchebichef vectors (v2 behaviour)
#         fold    : per-feature z-score, mean/std fitted on the outer TRAINING fold
#                   only, applied to train, inner-val and test (towers AND skip)
#         subject : per-feature z-score within each student-session, label-free
#                   (removes between-student offsets: the model can only use
#                   within-student variation; transductive, declared as such)
#   folds session      : v2 folds (student-sessions dealt round-robin)
#         person_strat : StratifiedGroupKFold over PERSONS (the same student in
#                        two sessions is never split), stratified on the label;
#                        the inner validation split is by person as well
NORM = "none"
FOLD_MODE = "session"
VARIANTS = os.environ.get("GTAO_VARIANTS",
                          "none:person_strat,fold:person_strat,subject:person_strat")
RUN_H4 = os.environ.get("GTAO_H4", "0") == "1"   # masked-stream probe (off in v3)

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
if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
print("device:", device, "| FAST:", FAST, "| fits:", (len(FUSIONS) * len(SKIPS)
      * len(MDROPS) * len(LOSSES) + 2 * len(STREAMS) + 1) * N_FOLDS * len(SEEDS))


# ---------------------------------------------------------------------------
# 2. Small helpers: progress log and the QWK metric
# ---------------------------------------------------------------------------
def log_step(step, **payload):
    """Record one progress step in grid_v1_metrics.json.

    Inputs: ``step`` is a name such as "corpus"; ``payload`` are any keyword
    values to store. The file is re-read every time so earlier content is kept.
    """
    if METRICS.exists():
        doc = json.loads(METRICS.read_text())
    else:
        doc = {}
    doc["status"] = "running"
    if "steps" not in doc:
        doc["steps"] = {}
    entry = {"ts": time.strftime("%Y-%m-%d %H:%M:%S")}
    entry.update(payload)
    doc["steps"][step] = entry
    METRICS.write_text(json.dumps(doc, indent=2))


def qwk(y_true, y_pred):
    """Quadratic weighted kappa between true and predicted class labels.

    QWK compares two ratings on an ordered scale: disagreeing by 2 classes is
    punished 4x as much as disagreeing by 1. Returns 0.0 when the score is not
    defined (fewer than 2 samples, a single true class, or a NaN result).
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


# ---------------------------------------------------------------------------
# 3. Tchebichef moments (feature compression)
# ---------------------------------------------------------------------------
# A signal of length N is projected onto the first K discrete Tchebichef
# polynomials. The basis is an (N, K) matrix with orthonormal columns, so
# ``signal @ basis`` gives K numbers that summarise the signal (low orders =
# slow trends). The polynomials are computed with 50-digit precision (mpmath)
# because the plain recurrence is numerically unstable in float64.

def tchebichef_norm_squared(n, N):
    """Squared norm rho(n, N) of the order-n Tchebichef polynomial on N points.

    rho(n, N) = N * (N^2 - 1^2) * (N^2 - 2^2) * ... * (N^2 - n^2) / (2n + 1)
    """
    result = mpmath.mpf(N)
    for k in range(1, n + 1):
        result *= N * N - k * k
    return result / (2 * n + 1)


def tchebichef_basis(N, K):
    """Return the orthonormal Tchebichef basis as a float32 array of shape (N, K).

    Uses the three-term recurrence
        t_0(x) = 1
        t_1(x) = 2x - N + 1
        t_n(x) = ((2n - 1)(2x - N + 1) t_{n-1}(x) - (n - 1)(N^2 - (n - 1)^2) t_{n-2}(x)) / n
    and then divides each polynomial by its norm.
    """
    with mpmath.workdps(50):
        # polynomials[n][x] = value of the order-n polynomial at point x
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
            norm = mpmath.sqrt(tchebichef_norm_squared(n, N))
            for x in range(N):
                basis_matrix[x, n] = float(polynomials[n][x] / norm)
    return basis_matrix


_BASIS_CACHE = {}


def cached_tchebichef_basis(N, K):
    """Same as tchebichef_basis(N, K) but computed only once per (N, K)."""
    key = (N, K)
    if key not in _BASIS_CACHE:
        _BASIS_CACHE[key] = tchebichef_basis(N, K)
    return _BASIS_CACHE[key]


def transform_2d(img, K):
    """2-D Tchebichef moments of a square image: (N, N) -> (K, K).

    B.T @ img compresses the rows, the second ``@ B`` compresses the columns.
    """
    basis_matrix = cached_tchebichef_basis(img.shape[-1], K)
    return (basis_matrix.T @ img) @ basis_matrix


def transform_1d(x, K):
    """1-D Tchebichef moments along the last axis: (..., N) -> (..., K)."""
    basis_matrix = cached_tchebichef_basis(x.shape[-1], K)
    return x @ basis_matrix


# ---------------------------------------------------------------------------
# 4. Reading one subject bundle (labels, per-frame metadata, watch sensors)
# ---------------------------------------------------------------------------
# Same conventions as the published pipelines: rater change points with the
# ``filter_nulls`` convention (entries without a value are skipped), the 5 -> 3
# class collapse, and a plurality vote at the middle of each window.

def parse_ts(s):
    """Convert a time stamp such as "10:00:05", "10-00-05-250000" or
    "10.00.05" to seconds since midnight (a float).

    The parts are hours, minutes, optional seconds and optional fraction of a
    second (written as microsecond digits).
    """
    s = str(s).strip().replace(".", ":").replace("-", ":").replace("_", ":")
    parts = []
    for piece in s.split(":"):
        if piece != "":
            parts.append(piece)
    hours = int(parts[0])
    minutes = int(parts[1])
    if len(parts) > 2:
        seconds = int(parts[2])
    else:
        seconds = 0
    if len(parts) > 3 and parts[3]:
        fraction = float(parts[3].ljust(6, "0")) / 1e6
    else:
        fraction = 0.0
    return hours * 3600 + minutes * 60 + seconds + fraction


HEAD_POSE_KEYS = ("pitch", "yaw", "roll")


def points_to_array(point_dicts):
    """Turn a list of {"x":..,"y":..,"z":..} dicts into a float32 (n, 3) array.

    Raises KeyError or TypeError if an entry is malformed (callers catch these).
    """
    rows = []
    for point in point_dicts:
        rows.append([point["x"], point["y"], point["z"]])
    return np.array(rows, dtype=np.float32)


def read_facemesh(meta):
    """Facemesh landmarks of one frame as a (478, 3) array, or None if missing."""
    person = meta.get("person")
    if not person:
        person = {}
    face = person.get("face")
    if not face:
        face = {}
    facemesh = face.get("facemesh")
    if not isinstance(facemesh, list) or not facemesh:
        return None
    try:
        points = points_to_array(facemesh)
    except (KeyError, TypeError):
        return None
    if points.ndim == 2 and points.shape[1] == 3:
        return points
    return None


def read_pose(meta):
    """Body-pose key points of one frame as an (n, 3) array, or None.

    (Kept from the published pipeline; not used by this grid.)
    """
    person = meta.get("person")
    if not person:
        person = {}
    pose = person.get("pose")
    if not isinstance(pose, list) or not pose:
        return None
    try:
        points = points_to_array(pose)
    except (KeyError, TypeError):
        return None
    if points.ndim == 2 and points.shape[1] == 3:
        return points
    return None


def read_head_pose(meta):
    """Head pose of one frame as float32 [pitch, yaw, roll], or None if missing.

    Looks first at meta["head_pose"], then at meta["person"]["face"]["headpose"]["pose"].
    """
    if isinstance(meta.get("head_pose"), dict):
        head_pose = meta.get("head_pose")
    else:
        head_pose = None
    if head_pose is None:
        try:
            head_pose = meta["person"]["face"]["headpose"]["pose"]
        except (KeyError, TypeError):
            return None
    if not isinstance(head_pose, dict):
        return None
    try:
        values = []
        for key in HEAD_POSE_KEYS:
            values.append(float(head_pose[key]))
        return np.array(values, dtype=np.float32)
    except (KeyError, TypeError, ValueError):
        return None


def norm_watch_channel(sensor, fields):
    """Map the raw field names of a watch sensor to standard channel names.

    Example: sensor "accelerometer" with fields value0, value1, value2 gives
    {"value0": "Accel.x", "value1": "Accel.y", "value2": "Accel.z"}.
    Unknown sensors map each field to "<sensor>.<field>".
    """
    sensor_lower = sensor.lower()
    if "hr" in sensor_lower or "heart" in sensor_lower:
        return {"value0": "HeartRate.bpm"}
    n_axes = min(len(fields), 4)
    if "accel" in sensor_lower:
        mapping = {}
        for i in range(n_axes):
            mapping[f"value{i}"] = f"Accel.{'xyzw'[i]}"
        return mapping
    if "gyr" in sensor_lower:
        mapping = {}
        for i in range(n_axes):
            mapping[f"value{i}"] = f"Gyro.{'xyzw'[i]}"
        return mapping
    if "rot" in sensor_lower:
        mapping = {}
        for i in range(n_axes):
            mapping[f"value{i}"] = f"Rot.{'wxyz'[i]}"
        return mapping
    if "light" in sensor_lower:
        return {"value0": "Light.lux"}
    mapping = {}
    for field in fields:
        mapping[field] = f"{sensor}.{field}"
    return mapping


def is_number(value):
    """True for int and float values (note: bool counts as int in Python)."""
    return isinstance(value, (int, float))


def append_sample(channels, channel_name, time_value, sample_value):
    """Add one (time, value) sample to ``channels[channel_name]``.

    ``channels`` maps a channel name to a pair of lists (times, values).
    """
    if channel_name not in channels:
        channels[channel_name] = ([], [])
    channel = channels[channel_name]
    channel[0].append(time_value)
    channel[1].append(float(sample_value))


def read_watch_sensors(watch_dir):
    """Read all smart-watch JSON files of a subject.

    Returns {channel name: (times float64 array, values float32 array)},
    each channel sorted by time. Two file layouts are supported:
    * new: {"data": {sensor: [{"<...time...>": t, "value0": v, ...}, ...]}}
    * legacy: {"Datos": {sensor: [{"time": t, <field>: v, ...}, ...]}}
    """
    channels = {}
    if not watch_dir.is_dir():
        return channels
    for file_path in sorted(watch_dir.iterdir()):
        if "__" in file_path.name or file_path.suffix != ".json":
            continue
        try:
            blob = json.loads(file_path.read_text())
        except json.JSONDecodeError:
            continue

        # --- new layout ---
        if isinstance(blob, dict):
            data = blob.get("data")
        else:
            data = None
        if isinstance(data, dict):
            for sensor, records in data.items():
                if not isinstance(records, list):
                    continue
                for sample in records:
                    if not isinstance(sample, dict):
                        continue
                    # the time key is the first key that contains "time"
                    time_key = None
                    for key in sample:
                        if "time" in key.lower():
                            time_key = key
                            break
                    if time_key is None:
                        continue
                    raw_time = sample[time_key]
                    if isinstance(raw_time, (int, float)):
                        time_value = raw_time
                    else:
                        time_value = parse_ts(raw_time)
                    numeric_fields = []
                    for key in sample:
                        if key != time_key and is_number(sample[key]):
                            numeric_fields.append(key)
                    fields = sorted(numeric_fields)
                    mapping = norm_watch_channel(str(sensor), fields)
                    for field in fields:
                        channel_name = mapping.get(field, f"{sensor}.{field}")
                        append_sample(channels, channel_name, time_value, sample[field])
            continue

        # --- legacy raw-download layout ---
        if isinstance(blob, dict):
            datos = blob.get("Datos")
        else:
            datos = None
        if not isinstance(datos, dict):
            continue
        for sensor_type, samples in datos.items():
            if not isinstance(samples, list):
                continue
            for sample in samples:
                if not isinstance(sample, dict):
                    continue
                time_key = None
                for key in sample:
                    if key.lower() == "time":
                        time_key = key
                        break
                if time_key is None:
                    continue
                time_value = sample[time_key]
                if not isinstance(time_value, (int, float)):
                    time_value = parse_ts(time_value)
                mapping = norm_watch_channel(str(sensor_type), list(sample.keys()))
                for field_name, value in sample.items():
                    if field_name == time_key or not isinstance(value, (int, float)):
                        continue
                    channel_name = mapping.get(field_name, f"{sensor_type}.{field_name}")
                    append_sample(channels, channel_name, time_value, value)

    result = {}
    for channel_name, (times, values) in channels.items():
        times_array = np.asarray(times, dtype=np.float64)
        order = np.argsort(times_array, kind="mergesort")    # stable sort by time
        result[channel_name] = (times_array[order],
                                np.asarray(values, dtype=np.float32)[order])
    return result


def change_point_time(change_point):
    """Sort key for a rater change point (time, value): its time."""
    return change_point[0]


def read_rater_labels(labels_dir):
    """Read every rater file of a subject.

    Returns {rater name: [(time in seconds, attention 1..5), ...]} sorted by
    time. Each entry is a "change point": the rater's score from that moment on.
    """
    raters = {}
    if not labels_dir.is_dir():
        return raters
    for file_path in sorted(labels_dir.iterdir()):
        if "__" in file_path.name or file_path.suffix != ".json":
            continue
        try:
            raw = json.loads(file_path.read_text())
        except json.JSONDecodeError:
            continue
        if not isinstance(raw, list):
            continue
        change_points = []
        for entry in raw:
            if not isinstance(entry, dict) or "datetime" not in entry:
                continue
            value = entry.get("attention")
            if value is None:
                continue
            change_points.append((parse_ts(entry["datetime"]), int(value)))
        raters[file_path.stem] = sorted(change_points, key=change_point_time)
    return raters


def value_at(change_points, t):
    """The rater's score in force at time ``t`` (last change point <= t), or None."""
    best = None
    for time_value, value in change_points:
        if time_value <= t:
            best = value
        else:
            break
    return best


def collapse_5_to_3(v):
    """Collapse the 1..5 attention scale to 3 classes: {1,2}->0, 3->1, {4,5}->2."""
    if v <= 2:
        return 0
    if v == 3:
        return 1
    return 2


def majority_vote(votes, n_classes=N_CLASSES):
    """Most frequent class among the non-None votes (ties -> lowest class)."""
    valid_votes = []
    for vote in votes:
        if vote is not None:
            valid_votes.append(vote)
    if not valid_votes:
        return None
    counts = np.bincount(np.asarray(valid_votes, dtype=int), minlength=n_classes)
    return int(counts.argmax())


# ---------------------------------------------------------------------------
# 5. Corpus encoding, one subject at a time
# ---------------------------------------------------------------------------
# Each bundle (~1 GB on Kaggle) is extracted to a temporary folder, turned into
# per-window per-stream Tchebichef feature vectors (+ plurality label + soft
# rater target), and then the extracted folder is deleted. Peak disk use is
# therefore one bundle.
#
# Soft target: per window, the share of raters voting for each of the 3 classes
# (used by the "soft" loss).

def pad_time(arr, k):
    """Make sure the time axis (axis 0) has at least ``k`` entries by
    repeating the last entry. Raises ValueError for an empty array."""
    if arr.shape[0] >= k:
        return arr
    if arr.shape[0] == 0:
        raise ValueError("empty time axis")
    repeated_last = np.repeat(arr[-1:], k - arr.shape[0], axis=0)
    return np.concatenate([arr, repeated_last], axis=0)


def subject_windows(subject_root, group_key):
    """All 4 s windows with plurality label + soft rater target for one subject.

    ``group_key`` = "group_01/experiment_01/subject_01", from the bundle filename
    (the extracted tree has no group/experiment nesting).

    Returns a list of dicts with keys: key, subject, t_start, t_end, label,
    soft, n_raters, frames (list of (time, image path)).
    Windows where no rater has a score, or that contain no frame, are skipped.
    """
    frames = []
    for image_path in sorted((subject_root / "images").glob("*.png")):
        try:
            frames.append((parse_ts(image_path.stem), image_path))
        except (ValueError, IndexError):
            continue
    if not frames:
        return []
    raters = read_rater_labels(subject_root / "labels")
    if not raters:
        return []
    frame_time_list = []
    for frame_time, _ in frames:
        frame_time_list.append(frame_time)
    frame_times = np.array(frame_time_list, dtype=np.float64)
    first_time = float(frame_times[0])
    last_time = float(frame_times[-1])
    windows = []
    t = first_time
    while t + WINDOW_S <= last_time + 1e-9:
        middle = t + WINDOW_S / 2
        # one collapsed vote per rater who has a score at the window middle
        votes = []
        for change_points in raters.values():
            if value_at(change_points, middle) is not None:
                votes.append(collapse_5_to_3(value_at(change_points, middle)))
        if not votes:
            t += WINDOW_S
            continue
        label = majority_vote(votes)
        low, high = np.searchsorted(frame_times, [t, t + WINDOW_S])
        frame_indices = list(range(int(low), int(high)))
        if not frame_indices:
            t += WINDOW_S
            continue
        group = group_key.split("/")[0]
        experiment = group_key.split("/")[1]
        subject = group_key.split("/")[2]
        soft = np.zeros(N_CLASSES, dtype=np.float32)
        for vote in votes:
            soft[vote] += 1.0
        soft /= soft.sum()
        window_frames = []
        for i in frame_indices:
            window_frames.append(frames[i])
        windows.append({
            "key": f"{group}/{experiment}/{subject}#{t:.2f}",
            "subject": f"{group}/{experiment}/{subject}",
            "t_start": t, "t_end": t + WINDOW_S,
            "label": label, "soft": soft, "n_raters": len(votes),
            "frames": window_frames})
        t += WINDOW_S
    return windows


def sequence_moments(sequence, n_values_per_step):
    """Compress a (time, ...) sequence along time with 1-D Tchebichef moments.

    The sequence is reshaped to (n_values_per_step, time) so every value gets
    its own time series, transformed to (n_values_per_step, SEQ_K) and then
    flattened to a 1-D vector of length n_values_per_step * SEQ_K.
    """
    time_last = np.moveaxis(sequence, 0, -1)
    per_value_series = time_last.reshape(n_values_per_step, sequence.shape[0])
    return transform_1d(per_value_series, SEQ_K).ravel()


def encode_windows(windows, subject_root, physio):
    """Per-window per-stream feature tensor; a stream absent in a window -> None.

    Feature sizes per window: face 16*16*6 = 1536, facemesh 478*3*6 = 8604,
    head_pose 3*6 = 18, physio 8*6 = 48.
    """
    rows = []
    for window in windows:
        faces = []
        facemesh_sequence = []
        head_pose_sequence = []
        for _, image_path in window["frames"]:
            stem = image_path.stem
            image = Image.open(image_path)
            grey_image = image.convert("L")
            small_image = grey_image.resize((FACE_SIZE, FACE_SIZE))
            pixels = np.asarray(small_image, dtype=np.float32) / 255.0
            faces.append(transform_2d(pixels, FACE_K))            # (16, 16)
            meta_path = subject_root / "metadata" / f"{stem}.json"
            if not meta_path.exists():
                continue
            try:
                meta = json.loads(meta_path.read_text())
            except json.JSONDecodeError:
                continue
            facemesh = read_facemesh(meta)
            if facemesh is not None:
                facemesh_sequence.append(facemesh)
            head_pose = read_head_pose(meta)
            if head_pose is not None:
                head_pose_sequence.append(head_pose)

        if not faces:
            continue
        features = {}
        face_stack = pad_time(np.stack(faces), SEQ_K)               # (time, 16, 16)
        features["face"] = sequence_moments(face_stack, -1)
        if facemesh_sequence:
            facemesh_stack = pad_time(np.stack(facemesh_sequence), SEQ_K)   # (time, 478, 3)
            features["facemesh"] = sequence_moments(facemesh_stack, 478 * 3)
        if head_pose_sequence:
            head_pose_stack = pad_time(np.stack(head_pose_sequence), SEQ_K)  # (time, 3)
            features["head_pose"] = sequence_moments(head_pose_stack, 3)

        # Physio: every channel must have at least one sample in the window.
        physio_ok = True
        channel_features = []
        for channel_name in PHYSIO_CHANNELS:
            channel = physio.get(channel_name)
            if channel is None:
                physio_ok = False
                break
            sample_times, sample_values = channel
            in_window = (sample_times >= window["t_start"]) & (sample_times < window["t_end"])
            x = sample_values[in_window]
            if len(x) < SEQ_K:
                if len(x) == 0:
                    physio_ok = False
                    break
                x = np.pad(x, (0, SEQ_K - len(x)), mode="edge")
            channel_features.append(transform_1d(x[None, :], SEQ_K).ravel())
        if physio_ok:
            features["physio"] = np.concatenate(channel_features)

        row = {}
        for key in ("key", "subject", "label", "soft", "n_raters"):
            row[key] = window[key]
        for stream in STREAMS:
            if stream != "physio":
                row[stream] = None
        row["physio"] = None
        for stream, vector in features.items():
            if stream in STREAMS:
                row[stream] = vector
        rows.append(row)
    return rows


def iter_subject_dirs(root, workdir):
    """Yield ``(subject_root, bundle_stem)`` for every per-subject zipbin.

    Each bundle is extracted into ``workdir/<stem>``; after its subjects have
    been processed the extracted folder is deleted again. Falls back to *.zip
    files if there are no *.zipbin files.
    """
    bundles = sorted(root.glob("*.zipbin"))
    if not bundles:
        bundles = sorted(root.glob("*.zip"))
    for bundle in bundles:
        target = workdir / bundle.stem
        if target.exists():
            shutil.rmtree(target, ignore_errors=True)
        target.mkdir(parents=True, exist_ok=True)
        try:
            with zipfile.ZipFile(bundle) as archive:
                archive.extractall(target)
        except zipfile.BadZipFile:
            shutil.rmtree(target, ignore_errors=True)
            continue
        stem = bundle.stem
        # a subject folder is any folder that has an "images" sub-folder
        subject_folders = []
        for images_path in target.rglob("images"):
            if images_path.is_dir():
                subject_folders.append(images_path.parent)
        for subject_folder in sorted(subject_folders):
            yield subject_folder, stem
        shutil.rmtree(target, ignore_errors=True)


def build_corpus(dataroot, workdir, cache, fast_max=None):
    """Encode every subject; cache the assembled arrays to ``cache``.

    Returns the loaded .npz file, which behaves like a dict with keys:
    keys, subjects, y (labels), soft (rater shares), n_raters,
    avail_<stream> (bool per window) and feat_<stream> (float32 matrix,
    NaN rows where the stream is missing; shape (0, 0) if never present).
    ``fast_max`` stops after that many subjects (smoke tests).
    """
    if cache.exists():
        print("loading cached corpus:", cache)
        return np.load(cache, allow_pickle=True)

    records = []
    seen = {}
    start_time = time.time()
    for subject_folder, stem in iter_subject_dirs(dataroot, workdir):
        group_key = "/".join(stem.split("__")[:3])
        try:
            windows = subject_windows(subject_folder, group_key)
            physio = read_watch_sensors(subject_folder / "watch_sensors")
        except Exception as error:
            print(f"  ! {stem}: {type(error).__name__}: {error}", flush=True)
            continue
        if not windows:
            print(f"  - {stem}: no labelled windows", flush=True)
            continue
        rows = encode_windows(windows, subject_folder, physio)
        if not rows:
            print(f"  - {stem}: face encode failed", flush=True)
            continue
        records.extend(rows)
        seen[stem] = len(rows)
        print(f"  + {stem}: {len(rows)} windows "
              f"[{group_key}] (elapsed {time.time() - start_time:.0f}s)", flush=True)
        if fast_max and len(seen) >= fast_max:
            break

    if not records:
        raise SystemExit("no windows collected")

    key_list = []
    subject_list = []
    label_list = []
    soft_list = []
    n_raters_list = []
    for record in records:
        key_list.append(record["key"])
        subject_list.append(record["subject"])
        label_list.append(record["label"])
        soft_list.append(record["soft"])
        n_raters_list.append(record["n_raters"])
    keys = np.asarray(key_list)
    subjects = np.asarray(subject_list)
    y = np.asarray(label_list, dtype=np.int64)
    soft = np.stack(soft_list).astype(np.float32)
    n_raters = np.asarray(n_raters_list, dtype=np.int64)

    avail = {}
    for stream in STREAMS:
        present_flags = []
        for record in records:
            present_flags.append(record[stream] is not None)
        avail[stream] = np.asarray(present_flags, dtype=bool)

    features = {}
    for stream in STREAMS:
        features[stream] = None
    for stream in STREAMS:
        present_rows = []
        for i, record in enumerate(records):
            if record[stream] is not None:
                present_rows.append(i)
        if present_rows:
            present_vectors = []
            for i in present_rows:
                present_vectors.append(records[i][stream])
            present_matrix = np.asarray(present_vectors, dtype=np.float32)
            full = np.full((len(records), present_matrix.shape[1]), np.nan, dtype=np.float32)
            full[present_rows] = present_matrix
            features[stream] = full
        denominator = len(records)
        if denominator == 0:
            denominator = 1
        print(f"stream {stream}: present in {len(present_rows)}/{len(records)} windows "
              f"({100 * len(present_rows) / denominator:.1f}%)")

    # A stream that never appears is stored as an empty (0, 0) matrix.
    empty2d = np.zeros((0, 0), dtype=np.float32)
    saved_features = {}
    for stream in STREAMS:
        if features[stream] is not None:
            saved_features[stream] = features[stream]
        else:
            saved_features[stream] = empty2d
    np.savez(cache, keys=keys, subjects=subjects, y=y, soft=soft,
             n_raters=n_raters, avail_face=avail["face"],
             avail_facemesh=avail["facemesh"], avail_head_pose=avail["head_pose"],
             avail_physio=avail["physio"],
             feat_face=saved_features["face"],
             feat_facemesh=saved_features["facemesh"],
             feat_head_pose=saved_features["head_pose"],
             feat_physio=saved_features["physio"])
    print(f"corpus: {len(records)} windows, {len(seen)} subjects, "
          f"{time.time() - start_time:.0f}s")
    return np.load(cache, allow_pickle=True)


def folder_has_zipbin(folder):
    """True if ``folder`` contains at least one *.zipbin file."""
    for _ in Path(folder).glob("*.zipbin"):
        return True
    return False


def locate_dataset_root():
    """Search /kaggle/input for the folder that holds the *.zipbin bundles.

    Returns a Path, or None if nothing is found (e.g. dataset not mounted yet).
    """
    for candidate in ("/kaggle/input/dipseer-paper1-slice",
                      "/kaggle/input/datasets/dipseer-paper1-slice"):
        if folder_has_zipbin(candidate):
            return Path(candidate)
    for folder_path, _, file_names in os.walk("/kaggle/input"):
        for file_name in file_names:
            if file_name.endswith(".zipbin"):
                return Path(folder_path)
    return None


# ---------------------------------------------------------------------------
# 6. The model: towers, gate vs CORE fusion, global skip, CORAL head
# ---------------------------------------------------------------------------
# Shapes used below: B = batch size, S = number of streams, E = EMB = 64.
#
# * gate fusion: cross-modal self-attention + a learned gate alpha (one weight
#   per stream, summing to 1) that can push a noisy stream to ~0 ("rejection").
# * core fusion: the CORE-style fusion operationalised on DIPSER -
#   self-attention over the stream embeddings followed by a plain mean. It can
#   re-weight through attention but cannot switch a stream off.
# * global skip: one linear map per stream from the raw Tchebichef vector
#   straight to the vector before the head; gives the input a short gradient path.
# * modality dropout: during training, whole tower outputs are zeroed with
#   probability MDROP_P so fusion learns to cope with missing streams.

class StreamTower(nn.Module):
    """Small MLP for one stream: in_dim -> HID -> EMB, followed by LayerNorm."""

    def __init__(self, in_dim, hidden=HID, emb=EMB):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, hidden), nn.GELU(),
                                 nn.Linear(hidden, emb), nn.LayerNorm(emb))

    def forward(self, x):
        """x: (B, in_dim) -> (B, EMB)."""
        return self.net(x)


class GatedCrossModalFusion(nn.Module):
    """Self-attention across streams, then a learned gate that weights them.

    Input z: (B, S, E). Output: fused (B, E) and alpha (B, S), the gate weights.
    """

    def __init__(self, n_streams, emb=EMB):
        super().__init__()
        self.proj_q = nn.Linear(emb, emb, bias=False)
        self.proj_k = nn.Linear(emb, emb, bias=False)
        self.gate_net = nn.Sequential(nn.Linear(n_streams * emb, n_streams * emb),
                                      nn.GELU(), nn.Linear(n_streams * emb, n_streams))

    def forward(self, z):
        """Scaled dot-product attention between streams, then gated sum."""
        q = self.proj_q(z)
        k = self.proj_k(z)
        # attn[b, i, j] = how much stream i looks at stream j; shape (B, S, S)
        attn = torch.softmax(q @ k.transpose(1, 2) / math.sqrt(z.shape[-1]), dim=-1)
        attended = attn @ z                                     # (B, S, E)
        # gate: all streams flattened -> one score per stream -> softmax
        flat = attended.reshape(attended.shape[0], -1)          # (B, S*E)
        alpha = torch.softmax(self.gate_net(flat), dim=-1)      # (B, S)
        fused = (alpha.unsqueeze(-1) * attended).sum(dim=1)     # (B, E)
        return fused, alpha


class CoreAttentionFusion(nn.Module):
    """CORE-style: self-attention then an unweighted mean pool (no rejection).

    Input z: (B, S, E). Output: fused (B, E) and None (there is no gate).
    """

    def __init__(self, n_streams, emb=EMB):
        super().__init__()
        self.proj_q = nn.Linear(emb, emb, bias=False)
        self.proj_k = nn.Linear(emb, emb, bias=False)

    def forward(self, z):
        """Same attention as the gate fusion, but every stream counts equally."""
        q = self.proj_q(z)
        k = self.proj_k(z)
        attn = torch.softmax(q @ k.transpose(1, 2) / math.sqrt(z.shape[-1]), dim=-1)
        attended = attn @ z                                     # (B, S, E)
        return attended.mean(dim=1), None


class GlobalSkip(nn.Module):
    """One bias-free linear map per stream (raw features -> EMB), summed."""

    def __init__(self, stream_dims, emb=EMB):
        super().__init__()
        projections = {}
        for stream, dim in stream_dims.items():
            projections[stream] = nn.Linear(dim, emb, bias=False)
        self.proj = nn.ModuleDict(projections)

    def forward(self, streams):
        """streams: {name: (B, dim)} -> (B, EMB), the sum over streams."""
        projected = []
        for stream, x in streams.items():
            projected.append(self.proj[stream](x))
        return torch.stack(projected).sum(dim=0)


class CORALHead(nn.Module):
    """Ordinal output head (CORAL: COnsistent RAnk Logits).

    For 3 ordered classes it outputs 2 logits: logit_k answers "is the class
    greater than k?". All logits share one weight vector and differ only by a
    bias, so the thresholds stay ordered. Class probabilities are differences
    of consecutive "greater than" probabilities.
    """

    def __init__(self, emb=EMB, num_classes=N_CLASSES):
        super().__init__()
        self.d = num_classes - 1                       # number of thresholds
        self.w = nn.Linear(emb, 1)
        self.b = nn.Parameter(torch.linspace(0.5, -0.5, self.d))

    def forward(self, x):
        """x: (B, EMB) -> logits (B, d); one shared score plus d biases."""
        return self.w(x) + self.b

    def probs(self, logits):
        """Logits (B, d) -> class probabilities (B, d + 1).

        P(y > k) = sigmoid(logit_k). With P(y > -1) = 1 and P(y > d) = 0,
        P(y = k) = P(y > k - 1) - P(y > k), clipped at 0 and renormalised.
        """
        prob_greater = torch.sigmoid(logits)
        extended = torch.cat([torch.ones_like(prob_greater[:, :1]), prob_greater,
                              torch.zeros_like(prob_greater[:, :1])], dim=1)
        p = (extended[:, :-1] - extended[:, 1:]).clamp(min=0)
        return p / p.sum(dim=1, keepdim=True)

    def loss(self, logits, y):
        """Binary cross-entropy against the targets [y > 0, y > 1, ...]."""
        k = torch.arange(self.d, device=logits.device)[None, :]
        targets = (y[:, None] > k).float()
        return F.binary_cross_entropy_with_logits(logits, targets)

    def predict(self, logits):
        """Most probable class for each row of logits, shape (B,)."""
        return self.probs(logits).argmax(dim=1)


def head_loss(head, logits, y, soft, loss="coral"):
    """Loss selector for the three factorial variants (same CORAL head).

    * "coral":  ordinal threshold BCE with the hard label y.
    * "soft":   threshold BCE against the soft rater shares: the target for
                threshold k is P(y > k) = 1 - (share of classes 0..k).
    * "hybrid": cross-entropy on the class probabilities
                + 0.5 * ordinal BCE
                + 0.1 * cross-entropy against a smoothed target that puts 0.7
                  on the true class and 0.15 on each neighbouring class.
    """
    if loss == "coral":
        return head.loss(logits, y)
    if loss == "soft":
        cumulative = torch.cumsum(soft, dim=1)[:, :head.d]
        return F.binary_cross_entropy_with_logits(logits, 1.0 - cumulative)
    # --- hybrid ---
    p = head.probs(logits)
    ce = -torch.log(p.gather(1, y[:, None]).squeeze(1) + 1e-8).mean()
    bce = head.loss(logits, y)
    smoothed_target = torch.zeros_like(p)
    smoothed_target.scatter_(1, y[:, None], 0.7)
    for offset in (-1, 1):
        neighbour = y + offset
        valid_rows = torch.nonzero((neighbour >= 0) & (neighbour < N_CLASSES)).squeeze(1)
        if valid_rows.numel():
            smoothed_target[valid_rows, neighbour[valid_rows].clamp(0, N_CLASSES - 1)] += 0.15
    smoothed_target = smoothed_target / smoothed_target.sum(dim=1, keepdim=True)
    smooth = -(smoothed_target * torch.log(p + 1e-8)).sum(dim=1).mean()
    return ce + 0.5 * bce + 0.1 * smooth


class GTAONetGrid(nn.Module):
    """The full model: one tower per stream -> fusion (+ optional skip) -> CORAL head.

    Note: the order in which sub-modules are created below fixes the random
    initial weights for a given seed; do not reorder it.
    """

    def __init__(self, stream_dims, fusion="gate", use_skip=True, mdrop=False,
                 emb=EMB):
        super().__init__()
        self.streams = sorted(stream_dims)
        towers = {}
        for stream in self.streams:
            towers[stream] = StreamTower(stream_dims[stream])
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
        self.input_norm = None          # v3: {stream: (mean, std)} or None

    def set_input_norm(self, stats):
        """Standardise every input as (x - mean) / std before the towers and the skip.

        Plain attributes (not buffers): load_state_dict never touches them.
        """
        self.input_norm = stats

    def forward(self, streams, apply_mdrop=None):
        """streams: {name: (B, dim)} -> (logits (B, 2), gate alpha (B, S) or None).

        Modality dropout runs when the model is training with mdrop=True, or
        when ``apply_mdrop`` is truthy.
        """
        if self.input_norm is not None:
            normalised = {}
            for stream, tensor in streams.items():
                mean, std = self.input_norm[stream]
                normalised[stream] = (tensor - mean) / std
            streams = normalised
        tower_outputs = []
        for stream in self.streams:
            tower_outputs.append(self.towers[stream](streams[stream]))
        z = torch.stack(tower_outputs, dim=1)                   # (B, S, E)
        if (self.training and self.mdrop) or apply_mdrop:
            # keep each stream of each sample with probability 1 - p
            keep_mask = torch.rand(z.shape[0], z.shape[1], device=z.device) > self.p
            z = z * keep_mask.unsqueeze(-1).float()
        fused, alpha = self.fusion(z)
        if self.use_skip and self.skip is not None:
            fused = fused + self.skip(streams)
        return self.head(fused), alpha


# ---------------------------------------------------------------------------
# 7. Training helpers
# ---------------------------------------------------------------------------
def to_torch(streams_dict):
    """Convert {name: numpy array} to {name: torch tensor on ``device``}."""
    tensors = {}
    for stream, array in streams_dict.items():
        tensors[stream] = torch.as_tensor(array, device=device)
    return tensors


def gather_stream_features(corpus, stream_names, row_indices):
    """{stream: corpus["feat_<stream>"][row_indices]} as torch tensors."""
    arrays = {}
    for stream in stream_names:
        arrays[stream] = corpus[f"feat_{stream}"][row_indices]
    return to_torch(arrays)


def select_streams(streams, stream_names):
    """New dict containing only ``stream_names`` (in that order) from ``streams``."""
    selected = {}
    for stream in stream_names:
        selected[stream] = streams[stream]
    return selected


def select_rows(streams, row_mask):
    """Apply the same boolean row mask to every stream tensor."""
    selected = {}
    for stream, tensor in streams.items():
        selected[stream] = tensor[row_mask]
    return selected


@torch.no_grad()
def evaluate(model, val_features, val_labels):
    """QWK, accuracy and mean absolute error of the model on one data set."""
    logits, _ = model(val_features, apply_mdrop=False)
    predictions = model.head.predict(logits).cpu().numpy()
    labels = val_labels.cpu().numpy()
    return {"qwk": qwk(labels, predictions),
            "acc": round(float((predictions == labels).mean()), 4),
            "mae": round(float(np.abs(predictions - labels).mean()), 4)}


def tower_first_grad_norm(model, streams, y, use_skip):
    """Gradient size reaching the first layer of every tower (H3 probe).

    Runs one CORAL loss backward pass with the skip path switched on or off
    and returns sqrt(sum of squared gradients) of all first-layer weights.
    """
    model.use_skip = use_skip
    model.zero_grad()
    logits, _ = model(streams, apply_mdrop=False)
    model.head.loss(logits, y).backward()
    total = 0.0
    for tower in model.towers.values():
        weight = tower.net[0].weight
        total += (weight.grad ** 2).sum().item()
    return math.sqrt(total)


def skip_grad_norm(model, streams, y):
    """Gradient size reaching the global-skip weights (H3 probe)."""
    model.use_skip = True
    model.zero_grad()
    logits, _ = model(streams, apply_mdrop=False)
    model.head.loss(logits, y).backward()
    total = 0.0
    for projection in model.skip.proj.values():
        total += (projection.weight.grad ** 2).sum().item()
    return math.sqrt(total)


def set_seed(seed):
    """Seed torch and numpy's global random generators."""
    torch.manual_seed(seed)
    np.random.seed(seed)


def inner_subject_split(train_features, train_labels, train_soft, train_subjects):
    """Split the training fold by subject: (1-INNER_FRAC) train, INNER_FRAC val.

    Inputs are the training-fold tensors and the subject id of every row.
    Draws random validation subjects (numpy global RNG) until both parts are
    non-empty and EACH part contains at least 2 classes (otherwise the inner
    validation QWK would be stuck at 0); gives up after 20 attempts.
    Returns (inner_train_features, inner_train_labels, inner_train_soft,
             inner_val_features, inner_val_labels, inner_val_subject_list),
    where the last item is the sorted list of inner validation subjects.
    """
    train_subjects = np.asarray(train_subjects).astype(str)
    unique_subjects = np.array(sorted(set(train_subjects.tolist())))
    n_val = max(1, int(round(INNER_FRAC * len(unique_subjects))))
    found = False
    for _ in range(20):                      # re-draw until both sides are usable
        permutation = np.random.permutation(len(unique_subjects))
        val_subjects = set(unique_subjects[permutation[:n_val]].tolist())
        is_val_list = []
        for subject in train_subjects:
            is_val_list.append(subject in val_subjects)
        is_val_np = np.array(is_val_list)
        has_train_rows = (~is_val_np).any()
        has_val_rows = is_val_np.any()
        if has_train_rows and has_val_rows:
            all_labels = train_labels.cpu().numpy()
            train_part_labels = all_labels[~is_val_np]
            val_part_labels = all_labels[is_val_np]
            train_part_ok = len(np.unique(train_part_labels)) >= 2
            val_part_ok = len(np.unique(val_part_labels)) >= 2
            if train_part_ok and val_part_ok:
                found = True
                break
    if not found:
        raise RuntimeError("could not draw a usable inner split")
    is_val = torch.as_tensor(is_val_np, device=train_labels.device)
    inner_train_features = select_rows(train_features, ~is_val)
    inner_train_labels = train_labels[~is_val]
    if train_soft is not None:
        inner_train_soft = train_soft[~is_val]
    else:
        inner_train_soft = None
    inner_val_features = select_rows(train_features, is_val)
    inner_val_labels = train_labels[is_val]
    inner_val_subject_list = sorted(val_subjects)
    return (inner_train_features, inner_train_labels, inner_train_soft,
            inner_val_features, inner_val_labels, inner_val_subject_list)


def fit_fold(model, train_features, train_labels, val_features, val_labels,
             stream_list, loss="coral", train_soft=None, subj_tr=None):
    """Train one fold, early stop on val QWK, return (best_state, gate_trace, metas).

    * With NESTED (v2 default) the validation set used for early stopping is an
      inner subject split of the training data; ``val_features``/``val_labels``
      (the outer test fold) are then not used here at all.
    * Without NESTED (v1) early stopping uses ``val_features``/``val_labels``.

    Returns:
      best  - {"qwk": best validation QWK, "state": weights, "epoch": its epoch}
      trace - every 5th epoch: validation QWK and the mean gate weights
      metas - {"epochs": epochs run, "loss_last": mean of the last 64 batch losses}
      inner - with NESTED: {"n_inner_train", "n_inner_val", "inner_val_subjects"}
              describing the inner split; without NESTED: None
    The model is left loaded with the best weights, in eval mode.
    """
    inner = None
    if NESTED:
        if subj_tr is None:
            raise ValueError("NESTED=1 requires subj_tr (training-fold subject ids)")
        (train_features, train_labels, train_soft,
         val_features, val_labels, inner_val_subjects) = inner_subject_split(
            train_features, train_labels, train_soft, subj_tr)
        inner = {"n_inner_train": int(len(train_labels)),
                 "n_inner_val": int(len(val_labels)),
                 "inner_val_subjects": inner_val_subjects}
        print(f"    inner split: train n={inner['n_inner_train']}, "
              f"val n={inner['n_inner_val']}, val subjects={inner_val_subjects}",
              flush=True)
    set_seed(int(np.random.randint(2 ** 31)))           # fold-rooted seed
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    best = {"qwk": -1.0, "state": None, "epoch": 0}
    trace = []
    batch_losses = []
    for epoch in range(1, EPOCHS + 1):
        model.train()
        order = torch.randperm(len(train_labels))
        for batch_start in range(0, len(train_labels), BATCH):
            batch_rows = order[batch_start:batch_start + BATCH]
            optimizer.zero_grad()
            batch_features = {}
            for stream in stream_list:
                batch_features[stream] = train_features[stream][batch_rows]
            logits, _ = model(batch_features)
            batch_labels = train_labels[batch_rows]
            if train_soft is not None:
                batch_soft = train_soft[batch_rows]
            else:
                batch_soft = None
            loss_value = head_loss(model.head, logits, batch_labels, batch_soft, loss)
            loss_value.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            batch_losses.append(loss_value.item())
        model.eval()
        val_scores = evaluate(model, val_features, val_labels)
        # ">= best" (with a tiny tolerance): later epochs win ties
        if val_scores["qwk"] > best["qwk"] - 1e-9:
            saved_state = {}
            for name, tensor in model.state_dict().items():
                saved_state[name] = tensor.detach().clone()
            best = {"qwk": val_scores["qwk"], "state": saved_state, "epoch": epoch}
        if epoch % 5 == 0:
            with torch.no_grad():
                _, alpha = model(select_streams(val_features, stream_list),
                                 apply_mdrop=False)
            gate = []
            if alpha is not None:
                for value in alpha.mean(dim=0).cpu().numpy():
                    gate.append(round(float(value), 4))
            trace.append({"epoch": epoch, "val_qwk": val_scores["qwk"], "gate": gate})
        if epoch >= MIN_EPOCHS and epoch - best["epoch"] >= PATIENCE:
            print(f"    early stop @ {epoch}, best val QWK {best['qwk']} "
                  f"(ep {best['epoch']})")
            break
    model.load_state_dict(best["state"])
    model.eval()
    metas = {"epochs": epoch, "loss_last": round(float(np.mean(batch_losses[-64:])), 4)}
    return best, trace, metas, inner


def complete_window_indices(corpus):
    """Indices of windows in which every available stream is present.

    Also returns the list of streams that exist in the corpus at all.
    """
    all_rows = np.arange(len(corpus["y"]))
    present_streams = []
    for stream in STREAMS:
        if corpus[f"feat_{stream}"].shape[0] > 0:
            present_streams.append(stream)
    keep = np.ones(len(all_rows), dtype=bool)
    for stream in present_streams:
        keep &= corpus[f"avail_{stream}"]
    return all_rows, all_rows[keep], present_streams


def subject_folds(subjects, seed):
    """Assign every window to a fold through its subject.

    The unique subjects are sorted, shuffled with ``np.random.default_rng(seed)``
    and dealt round-robin into N_FOLDS folds. Returns (fold of every window,
    the generator), because the generator is used again later (shuffle control).
    """
    unique_subjects = sorted(set(subjects.tolist()))
    rng = np.random.default_rng(seed)
    rng.shuffle(unique_subjects)
    fold_of_subject = {}
    for position, subject in enumerate(unique_subjects):
        fold_of_subject[subject] = position % N_FOLDS
    fold_list = []
    for subject in subjects:
        fold_list.append(fold_of_subject[subject])
    return np.array(fold_list), rng


def person_of(session_subject):
    """'group_01/experiment_05/subject_07' -> 'group_01/subject_07'.

    In DIPSER subject_YY is the same student in every experiment of a group.
    """
    parts = str(session_subject).split("/")
    if len(parts) == 3:
        return f"{parts[0]}/{parts[2]}"
    return str(session_subject)


def grouping_ids(subjects):
    """Grouping unit for the folds and the inner split under the current FOLD_MODE."""
    if FOLD_MODE == "person_strat":
        return np.array([person_of(s) for s in subjects])
    return np.asarray(subjects).astype(str)


def stratified_person_folds(corpus, idx_complete, labels, seed):
    """Fold of every row (-1 outside ``idx_complete``) from StratifiedGroupKFold.

    Groups are persons, strata are the labels of the complete-window population.
    Returns (fold of every row, generator) like ``subject_folds``.
    """
    from sklearn.model_selection import StratifiedGroupKFold
    fold_of = np.full(len(corpus["y"]), -1, dtype=int)
    groups = grouping_ids(corpus["subjects"][idx_complete])
    splitter = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
    for fold, (_, test_part) in enumerate(
            splitter.split(np.zeros(len(idx_complete)), labels[idx_complete], groups)):
        fold_of[idx_complete[test_part]] = fold
    return fold_of, np.random.default_rng(seed)


def subject_standardise(corpus):
    """Per-feature z-score within each student-session (label-free).

    Statistics come from that session's complete windows only; constant features
    become 0. Returns a plain dict copy of the corpus with the features replaced.
    """
    out = {}
    for name in corpus.files if hasattr(corpus, "files") else corpus.keys():
        out[name] = corpus[name]
    _, idx_complete, present = complete_window_indices(out)
    subjects = np.asarray(out["subjects"]).astype(str)
    for stream in present:
        features = out[f"feat_{stream}"].astype(np.float32).copy()
        for subject in np.unique(subjects[idx_complete]):
            rows = idx_complete[subjects[idx_complete] == subject]
            block = features[rows]
            mean = block.mean(axis=0)
            std = block.std(axis=0)
            std[std < 1e-8] = np.inf                   # constant feature -> 0
            features[rows] = (block - mean) / std
        out[f"feat_{stream}"] = features
    return out


def fold_norm_stats(train_features):
    """Per-stream (mean, std) of the training-fold features, for model.set_input_norm."""
    stats = {}
    for stream, tensor in train_features.items():
        mean = tensor.mean(dim=0)
        std = tensor.std(dim=0, unbiased=False)
        std = torch.where(std < 1e-8, torch.ones_like(std), std)
        stats[stream] = (mean, std)
    return stats


def decomposition(labels, predictions, subjects):
    """Extra v3 diagnostics on the pooled out-of-fold predictions.

    * between_r: correlation of per-subject mean label and mean prediction
      (does the model rank STUDENTS?);
    * within_r: correlation after removing each subject's means
      (does it follow attention changes WITHIN a student?);
    * balanced accuracy and per-class recall.
    """
    labels = np.asarray(labels, dtype=float)
    predictions = np.asarray(predictions, dtype=float)
    subjects = np.asarray(subjects).astype(str)
    unique = sorted(set(subjects.tolist()))
    mean_y = np.array([labels[subjects == s].mean() for s in unique])
    mean_p = np.array([predictions[subjects == s].mean() for s in unique])
    y_dev = np.concatenate([labels[subjects == s] - labels[subjects == s].mean() for s in unique])
    p_dev = np.concatenate([predictions[subjects == s] - predictions[subjects == s].mean()
                            for s in unique])

    def corr(a, b):
        if a.std() < 1e-12 or b.std() < 1e-12:
            return 0.0
        return round(float(np.corrcoef(a, b)[0, 1]), 4)

    recalls = []
    for k in range(N_CLASSES):
        mask = labels == k
        recalls.append(round(float((predictions[mask] == k).mean()), 4) if mask.any() else None)
    present = [r for r in recalls if r is not None]
    return {"between_r": corr(mean_y, mean_p), "within_r": corr(y_dev, p_dev),
            "balanced_acc": round(float(np.mean(present)), 4) if present else None,
            "recall": recalls}


def pooled_scores(labels, predictions):
    """QWK / accuracy / MAE over all out-of-fold predictions (rounded)."""
    return {"qwk": round(qwk(labels, predictions), 4),
            "acc": round(float(np.mean(predictions == labels)), 4),
            "mae": round(float(np.mean(np.abs(predictions - labels))), 4)}


# ---------------------------------------------------------------------------
# 8. H4 probe: masked evaluation (stream-drop robustness)
# ---------------------------------------------------------------------------
# gTAO-Net claims that its gate can reject a broken/noisy stream while CORE
# fusion cannot. H4 tests this at inference time: re-train the two focal
# configs (gate_skip1_drop0_soft and core_skip1_drop0_soft), then at test time
# zero out one stream at a time ("the sensor died") and measure how much the
# pooled QWK falls. If the hypothesis holds, gate should fall less than core and
# its alpha should move weight to the surviving streams.

MASK_CFGS = (
    {"fusion": "gate", "use_skip": True, "mdrop": False, "loss": "soft"},
    {"fusion": "core", "use_skip": True, "mdrop": False, "loss": "soft"},
)


def run_masked_eval(corpus, seeds=(0,)):
    """Train the focal configs, evaluate with each stream zero-masked at eval time.

    Returns {config name: {condition: {"qwk"/"acc"/"mae": [one value per seed]}}}
    where condition is "none" or "mask_<stream>".
    """
    all_rows, idx_complete, all_present = complete_window_indices(corpus)
    subjects = corpus["subjects"]
    ys = corpus["y"]
    soft_full = corpus["soft"]
    print(f"H4 population: {len(idx_complete)}/{len(all_rows)} complete windows")

    conditions = ["none"]
    for stream in all_present:
        conditions.append(f"mask_{stream}")

    per_seed = {}
    for seed in seeds:
        if FAST and seed != 0:
            continue
        fold_of, _ = subject_folds(subjects, seed)
        results_by_config = {}

        for cfg in MASK_CFGS:
            name = cfg_name(cfg)
            print(f"\n=== H4 masked-eval: {name} seed {seed} ===", flush=True)
            dims = {}
            for stream in all_present:
                dims[stream] = int(corpus[f"feat_{stream}"].shape[1])
            # out-of-fold predictions per condition (NaN = window not predicted)
            condition_predictions = {}
            for condition in conditions:
                condition_predictions[condition] = np.full(len(all_rows), np.nan)
            alpha_batches = {}
            for condition in condition_predictions:
                alpha_batches[condition] = []
            for fold in range(N_FOLDS):
                is_train = fold_of[idx_complete] != fold
                is_test = fold_of[idx_complete] == fold
                if len(set(ys[idx_complete][is_train])) < 2 or not is_test.any():
                    continue
                train_rows = idx_complete[is_train]
                test_rows = idx_complete[is_test]
                train_features = gather_stream_features(corpus, all_present, train_rows)
                train_labels = torch.as_tensor(ys[train_rows], device=device)
                train_soft = torch.as_tensor(soft_full[train_rows], device=device)
                test_features = gather_stream_features(corpus, all_present, test_rows)
                test_labels = torch.as_tensor(ys[test_rows], device=device)

                set_seed(int(f"{seed}{fold}"))
                model = GTAONetGrid(dims, fusion=cfg["fusion"],
                                    use_skip=cfg["use_skip"],
                                    mdrop=cfg["mdrop"]).to(device)
                best, _, metas, _ = fit_fold(model, train_features, train_labels,
                                          test_features, test_labels,
                                          subj_tr=grouping_ids(corpus["subjects"][train_rows]),
                                          stream_list=all_present,
                                          loss=cfg["loss"], train_soft=train_soft)
                model.load_state_dict(best["state"])
                model.eval()
                with torch.no_grad():
                    logits, alpha = model(test_features, apply_mdrop=False)
                    condition_predictions["none"][test_rows] = (
                        model.head.predict(logits).cpu().numpy())
                    if alpha is not None:
                        alpha_batches["none"].append(alpha.cpu().numpy())
                    for masked_stream in all_present:
                        # same test data, but one stream replaced by zeros
                        masked_features = {}
                        for stream, tensor in test_features.items():
                            if stream == masked_stream:
                                masked_features[stream] = torch.zeros_like(tensor)
                            else:
                                masked_features[stream] = tensor
                        logits_masked, alpha_masked = model(masked_features,
                                                            apply_mdrop=False)
                        condition_predictions[f"mask_{masked_stream}"][test_rows] = (
                            model.head.predict(logits_masked).cpu().numpy())
                        if alpha_masked is not None:
                            alpha_batches[f"mask_{masked_stream}"].append(
                                alpha_masked.cpu().numpy())
                print(f"  fold {fold}: fit ep={metas['epochs']} "
                      f"val QWK={best['qwk']:.3f}", flush=True)

            result = {"config": cfg, "seed": seed, "conditions": {}}
            for condition, predictions in condition_predictions.items():
                predicted_mask = np.isfinite(predictions)
                labels = ys[predicted_mask].astype(int)
                predicted = predictions[predicted_mask].astype(int)
                entry = pooled_scores(labels, predicted)
                entry["n"] = int(predicted_mask.sum())
                if alpha_batches[condition]:
                    all_alpha = np.concatenate(alpha_batches[condition], axis=0)
                    mean_alpha = []
                    for value in all_alpha.mean(axis=0):
                        mean_alpha.append(round(float(value), 4))
                    entry["gate_alpha"] = mean_alpha
                result["conditions"][condition] = entry
            results_by_config[name] = result
        per_seed[seed] = results_by_config

    print("\n=== H4 SUMMARY: pooled QWK by eval condition (mean±std over seeds) ===")
    available_seeds = sorted(per_seed)
    names = []
    for cfg in MASK_CFGS:
        names.append(cfg_name(cfg))
    header = f"{'config':26s}"
    for condition in conditions:
        header += f" {condition:>14s}"
    header += f" {'drop(max mask)':>15s}"
    print(header)
    summary = {}
    for name in names:
        line = f"{name:26s}"
        for condition in conditions:
            seed_qwks = []
            for seed in available_seeds:
                seed_qwks.append(per_seed[seed][name]["conditions"][condition]["qwk"])
            line += f" {np.mean(seed_qwks):+.4f}±{np.std(seed_qwks):.3f}"
        base_qwks = []
        for seed in available_seeds:
            base_qwks.append(per_seed[seed][name]["conditions"]["none"]["qwk"])
        base = np.mean(base_qwks)
        masks = []
        for stream in all_present:
            masks.append("mask_" + stream)
        drops = []
        for mask in masks:
            masked_qwks = []
            for seed in available_seeds:
                masked_qwks.append(per_seed[seed][name]["conditions"][mask]["qwk"])
            drops.append(base - np.mean(masked_qwks))
        line += f" {max(drops):>15.3f}"
        print(line)
        summary[name] = {}
        for condition in conditions:
            summary[name][condition] = {}
            for metric in ("qwk", "acc", "mae"):
                values = []
                for seed in available_seeds:
                    values.append(per_seed[seed][name]["conditions"][condition].get(metric))
                summary[name][condition][metric] = values
        alpha_seed0 = per_seed[0][name]["conditions"]["none"].get("gate_alpha")
        if alpha_seed0:
            print(f"  gate α (none)   : {alpha_seed0}")
            for mask in masks:
                print(f"  gate α ({mask:>10s}) : "
                      f"{per_seed[0][name]['conditions'][mask].get('gate_alpha')}")
    return summary


# ---------------------------------------------------------------------------
# 9. The main grid: one configuration x one seed x 5 folds
# ---------------------------------------------------------------------------
def cfg_name(cfg):
    """Short config name used for folders and tables.

    Examples: "gate_skip1_drop0" (coral loss), "core_skip0_drop1_soft",
    "unimodal_face_cw" (coral), "unimodal_face_cw_soft", "shuffle_core_skip1_drop1".
    """
    if cfg.get("unimodal"):
        if cfg.get("loss", "coral") == "soft":
            return f"unimodal_{cfg['unimodal']}_cw_soft"
        return f"unimodal_{cfg['unimodal']}_cw"
    if cfg.get("shuffle"):
        return "shuffle_core_skip1_drop1"
    base = f"{cfg['fusion']}_skip{int(cfg['use_skip'])}_drop{int(cfg['mdrop'])}"
    if cfg.get("loss", "coral") == "coral":
        return base
    return base + f"_{cfg['loss']}"


def has_empty_stream(corpus, stream_list):
    """True if any stream in ``stream_list`` has no feature rows in the corpus."""
    for stream in stream_list:
        if corpus[f"feat_{stream}"].shape[0] == 0:
            return True
    return False


def run_grid(corpus, seed, configs):
    """Run every config for one seed with subject-grouped 5-fold CV.

    For each config, writes ``RUNS/<config>/seed<seed>/oof.json`` (skipped if it
    already exists, so an interrupted run can resume). All configs use the same
    population: the windows where every stream is present.
    """
    all_rows = np.arange(len(corpus["y"]))
    ys = corpus["y"]
    soft_full = corpus["soft"]

    if FOLD_MODE == "person_strat":
        _, idx_population, _ = complete_window_indices(corpus)
        fold_of, rng = stratified_person_folds(corpus, idx_population, ys, seed)
    else:
        fold_of, rng = subject_folds(corpus["subjects"], seed)

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
            stream_list = []
            for stream in STREAMS:
                if corpus[f"feat_{stream}"].shape[0] > 0:
                    stream_list.append(stream)
        dims = {}
        for stream in stream_list:
            feature_matrix = corpus[f"feat_{stream}"]
            if feature_matrix.shape[0] > 0:
                dims[stream] = feature_matrix.shape[1]
        if not dims or has_empty_stream(corpus, stream_list):
            print(f"  skipped {name} (empty stream features)")
            continue

        # Population: complete windows (all streams present) for EVERY config - the
        # fusion grid, unimodal baselines and shuffle control all share this mask.
        _, idx_complete, _ = complete_window_indices(corpus)
        print(f"complete windows: {len(idx_complete)}/{len(all_rows)}  "
              f"class dist: {np.bincount(ys[idx_complete], minlength=N_CLASSES)}")

        y_run = ys.copy()
        soft_run = soft_full.copy()
        if cfg.get("shuffle"):
            # shuffle control: same population, labels randomly permuted
            permutation = rng.permutation(len(idx_complete))
            y_run[idx_complete] = ys[idx_complete][permutation]
            soft_run[idx_complete] = soft_full[idx_complete][permutation]
            print("shuffle control: labels permuted within complete population")

        oof_pred = np.full(len(all_rows), np.nan)       # out-of-fold predictions
        gate_trace_by_fold = {}
        for fold in range(N_FOLDS):
            is_train = fold_of[idx_complete] != fold
            is_test = fold_of[idx_complete] == fold
            if len(set(y_run[idx_complete][is_train])) < 2 or not is_test.any():
                continue
            train_rows = idx_complete[is_train]
            test_rows = idx_complete[is_test]
            train_features = gather_stream_features(corpus, stream_list, train_rows)
            train_labels = torch.as_tensor(y_run[train_rows], device=device)
            if cfg.get("loss") == "soft":
                train_soft = torch.as_tensor(soft_run[train_rows], device=device)
            else:
                train_soft = None
            test_features = gather_stream_features(corpus, stream_list, test_rows)
            test_labels = torch.as_tensor(y_run[test_rows], device=device)

            set_seed(int(f"{seed}{fold}"))
            if cfg.get("unimodal"):
                model = GTAONetGrid(dims, fusion="core", use_skip=False,
                                    mdrop=False).to(device)
            else:
                model = GTAONetGrid(dims, fusion=cfg["fusion"],
                                    use_skip=cfg["use_skip"],
                                    mdrop=cfg["mdrop"]).to(device)
            if NORM == "fold":
                model.set_input_norm(fold_norm_stats(train_features))

            # H3 probe (fold 0 only, skip configs): gradient sizes at initialisation
            if fold == 0 and out["probe"] is None and cfg.get("use_skip"):
                n_probe = min(64, len(train_rows))
                probe_features = gather_stream_features(corpus, stream_list,
                                                        train_rows[:n_probe])
                probe_labels = torch.as_tensor(y_run[train_rows[:n_probe]], device=device)
                grad_skip_on = tower_first_grad_norm(model, probe_features, probe_labels,
                                                     use_skip=True)
                grad_skip_off = tower_first_grad_norm(model, probe_features, probe_labels,
                                                      use_skip=False)
                grad_skip_path = skip_grad_norm(model, probe_features, probe_labels)
                model.use_skip = True
                out["probe"] = {"tower_first_skipON": round(grad_skip_on, 4),
                                "tower_first_skipOFF": round(grad_skip_off, 4),
                                "skip_grad": round(grad_skip_path, 4),
                                "ratio_skip_over_towerOFF":
                                    round(grad_skip_path / max(grad_skip_off, 1e-9), 3)}

            fold_start = time.time()
            best, trace, metas, inner = fit_fold(model, train_features, train_labels,
                                          test_features, test_labels,
                                          subj_tr=grouping_ids(corpus["subjects"][train_rows]),
                                          stream_list=stream_list,
                                          loss=cfg.get("loss", "coral"),
                                          train_soft=train_soft)
            model.load_state_dict(best["state"])
            with torch.no_grad():
                logits, _ = model(test_features, apply_mdrop=False)
                predictions = model.head.predict(logits).cpu().numpy()
            oof_pred[test_rows] = predictions
            gate_trace_by_fold[fold] = trace
            val_class_dist = []
            for count in np.bincount(y_run[test_rows], minlength=N_CLASSES):
                val_class_dist.append(int(count))
            fold_entry = {"fold": fold, "n_train": int(is_train.sum()),
                          "n_val": int(is_test.sum()),
                          "best_val_qwk": best["qwk"], "best_epoch": best["epoch"],
                          "val_class_dist": val_class_dist}
            for meta_key, meta_value in metas.items():
                fold_entry[meta_key] = meta_value
            fold_entry["wall_s"] = round(time.time() - fold_start, 1)
            if inner is not None:
                # nested mode only: describe the inner split used for early stopping
                fold_entry["n_inner_train"] = inner["n_inner_train"]
                fold_entry["n_inner_val"] = inner["n_inner_val"]
                fold_entry["inner_val_subjects"] = inner["inner_val_subjects"]
            out["folds"].append(fold_entry)
            print(f"  fold {fold}: val n={int(is_test.sum())} best QWK={best['qwk']:.4f} "
                  f"ep={best['epoch']}  [{round(time.time() - fold_start, 1)}s]", flush=True)

        predicted_mask = np.isfinite(oof_pred)
        labels = y_run[predicted_mask].astype(int)
        predicted = oof_pred[predicted_mask].astype(int)
        out["oof"] = {"keys": corpus["keys"][predicted_mask].astype(str).tolist(),
                      "y": labels.tolist(), "pred": predicted.tolist(),
                      "subject": corpus["subjects"][predicted_mask].astype(str).tolist()}
        out["pooled"] = pooled_scores(labels, predicted)
        out["pooled"].update(decomposition(labels, predicted,
                                           corpus["subjects"][predicted_mask]))
        out["pooled"]["norm"] = NORM
        out["pooled"]["folds"] = FOLD_MODE
        out["pooled"]["n"] = int(predicted_mask.sum())
        out["pooled"]["n_subjects"] = len(set(corpus["subjects"][predicted_mask].tolist()))
        gate_trace_json = {}
        for fold, trace in sorted(gate_trace_by_fold.items()):
            gate_trace_json[str(fold)] = trace
        out["gate_trace"] = gate_trace_json
        finish.write_text(json.dumps(out, indent=1))
        print(f"  POOLED QWK = {out['pooled']['qwk']} (n={predicted_mask.sum()}, "
              f"subjects={out['pooled']['n_subjects']})", flush=True)


# ---------------------------------------------------------------------------
# 10. Summary table
# ---------------------------------------------------------------------------
def summary_sort_key(row):
    """Sort rows by mean QWK (highest first), then by config name."""
    return (-row["qwk_mean"], row["config"])


def summarize(seeds, configs, write_to=METRICS):
    """Collect all oof.json files into one table, print it and save it.

    The table goes to ``write_to`` under the key "summary" (rows sorted by mean
    QWK). Returns the list of rows.
    """
    rows = []
    for cfg in configs:
        name = cfg_name(cfg)
        run_dir = RUNS / name
        qwks = []
        accs = []
        maes = []
        ns = []
        probes = []
        gates = []
        for seed in seeds:
            oof_file = run_dir / f"seed{seed}" / "oof.json"
            if not oof_file.exists():
                continue
            result = json.loads(oof_file.read_text())
            qwks.append(result["pooled"]["qwk"])
            accs.append(result["pooled"]["acc"])
            maes.append(result["pooled"]["mae"])
            ns.append(result["pooled"]["n"])
            probes.append(result.get("probe"))
            gates.append(result.get("gate_trace"))
        if not qwks:
            continue
        mean = float(np.mean(qwks))
        if len(qwks) > 1:
            std = float(np.std(qwks))
        else:
            std = 0.0
        if cfg.get("unimodal"):
            fusion = "unimodal"
        else:
            fusion = cfg["fusion"]
        rounded_qwks = []
        for value in qwks:
            rounded_qwks.append(round(value, 4))
        if probes and probes[0]:
            probe_seed0 = probes[0]
        else:
            probe_seed0 = None
        row = {"config": name, "fusion": fusion,
               "skip": cfg.get("use_skip"), "mdrop": cfg.get("mdrop"),
               "loss": cfg.get("loss", "coral"), "unimodal": cfg.get("unimodal"),
               "shuffle": cfg.get("shuffle"),
               "qwk_mean": round(mean, 4), "qwk_std": round(std, 4),
               "qwk_per_seed": rounded_qwks,
               "acc_mean": round(float(np.mean(accs)), 4),
               "mae_mean": round(float(np.mean(maes)), 4),
               "n_median": int(sorted(ns)[len(ns) // 2]),
               "probe_seed0": probe_seed0}
        # gate alphas at the last traced epoch, first seed, first fold (sorted by key)
        if gates and gates[0]:
            first_fold_trace = sorted(gates[0].items())[0][1]
            if first_fold_trace:
                row["gate_alpha_final"] = first_fold_trace[-1]["gate"]
        rows.append(row)

    rows.sort(key=summary_sort_key)
    print("\n=== GRID v1 SUMMARY (full 3-loss factorial, subject-disjoint) ===")
    print(f"{'config':26s} {'QWK mean±std':>14s} {'acc':>7s} {'mae':>6s} "
          f"{'n':>5s}  probe(skip/tower)  gate_alpha")
    for row in rows:
        probe = row["probe_seed0"]
        if probe:
            probe_text = f"{probe['ratio_skip_over_towerOFF']:>10.1f}"
        else:
            probe_text = "-"
        gate_alpha = row.get("gate_alpha_final")
        if gate_alpha:
            parts = []
            for value in gate_alpha:
                parts.append(f"{value:.2f}")
            gate_text = "[" + ", ".join(parts) + "]"
        else:
            gate_text = "-"
        print(f"{row['config']:26s} {row['qwk_mean']:+.4f}±{row['qwk_std']:.3f} "
              f"{row['acc_mean']:7.3f} {row['mae_mean']:6.3f} {row['n_median']:5d} "
              f"{probe_text}  {gate_text}")
    print(f"\nanchors: face ch.4 = {FACE_ANCHOR_CH4}, LORO ceiling = {LORO_CEILING}, "
          f"majority = 0.0")

    # Shuffle control: the largest |QWK| among shuffled-label rows.
    largest_shuffle_qwk = None
    for row in rows:
        if row.get("shuffle"):
            absolute_qwk = abs(row["qwk_mean"])
            if largest_shuffle_qwk is None or absolute_qwk > largest_shuffle_qwk:
                largest_shuffle_qwk = absolute_qwk
    if largest_shuffle_qwk is not None:
        if largest_shuffle_qwk <= SHUFFLE_TOL:
            verdict = "PASS"
        else:
            verdict = "FAIL"
        print(f"\nshuffle control: max |QWK| = {largest_shuffle_qwk:.4f} "
              f"({verdict} ≤ {SHUFFLE_TOL})")

    # Best unimodal baseline (first one wins on ties).
    best_unimodal = None
    for row in rows:
        if row.get("unimodal"):
            if best_unimodal is None or row["qwk_mean"] > best_unimodal["qwk_mean"]:
                best_unimodal = row
    if best_unimodal is not None:
        print(f"best in-protocol unimodal: {best_unimodal['config']} = "
              f"{best_unimodal['qwk_mean']:.4f}")

    if write_to.exists():
        doc = json.loads(write_to.read_text())
    else:
        doc = {}
    doc["status"] = "summary"
    doc["summary"] = {
        "grid": "gtao_grid_v1 full 3-loss factorial",
        "n_fits": len(configs) * N_FOLDS * len(seeds),
        "rows": rows,
        "anchors": {"face_ch4": FACE_ANCHOR_CH4, "loro_ceiling": LORO_CEILING,
                    "majority_qwk": 0.0}}
    write_to.write_text(json.dumps(doc, indent=2))
    return rows


# ---------------------------------------------------------------------------
# 11. Main
# ---------------------------------------------------------------------------
def build_configs():
    """The list of all experiment configurations, in a fixed order.

    24 fusion configs (fusion x skip x modality dropout x loss), then the 4
    unimodal CORAL baselines, then the shuffle control, then the 4 unimodal
    soft-target baselines. The soft baselines come LAST on purpose: the shuffle
    control draws from the fold random generator, so anything added before it
    would change its permutation.
    """
    configs = []
    for fusion in FUSIONS:
        for use_skip in SKIPS:
            for mdrop in MDROPS:
                for loss in LOSSES:
                    configs.append({"fusion": fusion, "use_skip": use_skip,
                                    "mdrop": mdrop, "loss": loss})
    for stream in STREAMS:
        configs.append({"unimodal": stream, "loss": "coral"})
    configs.append({"shuffle": True, "fusion": "core", "use_skip": True,
                    "mdrop": True, "loss": "coral"})
    for stream in STREAMS:
        configs.append({"unimodal": stream, "loss": "soft"})
    return configs


def find_data_root(start_time):
    """Decide where the *.zipbin bundles are.

    Kaggle: the attached dataset (waits up to 10 minutes for it to mount).
    Locally: DIPSEER_DATA_ROOT (or the Kaggle path if it happens to exist).
    """
    if ON_KAGGLE:
        dataroot = INPUT_DIR
    else:
        dataroot = Path(
            os.environ.get("DIPSEER_DATA_ROOT",
                           "/Users/mks/projects/research/phd/dipseer/kaggle"
                           "/paper2f/_synth_tmp/synth_slice"))
    if not ON_KAGGLE:
        for candidate in (INPUT_DIR, dataroot):
            if candidate.is_dir() and folder_has_zipbin(candidate):
                dataroot = candidate
                break
    if dataroot.is_dir():
        print("data root:", dataroot)
    else:
        dataroot = locate_dataset_root()
        while dataroot is None and time.time() - start_time < 600:
            print("waiting for dataset mount ...")
            time.sleep(5)
            dataroot = locate_dataset_root()
    if dataroot is None:
        raise FileNotFoundError("no *.zipbin found")
    return dataroot


def find_cached_corpus():
    """v3 re-uses the v2 feature cache (identical features, no 38-min extraction).

    On Kaggle the v2 kernel is attached as a kernel source; locally set
    GTAO_CORPUS to a corpus_grid_v1.npz (e.g. one built by pc_pipeline).
    """
    explicit = os.environ.get("GTAO_CORPUS")
    if explicit and Path(explicit).is_file():
        return Path(explicit)
    if ON_KAGGLE:
        for folder_path, _, file_names in os.walk("/kaggle/input"):
            if "corpus_grid_v1.npz" in file_names:
                return Path(folder_path) / "corpus_grid_v1.npz"
    return None


def main():
    """Load (or encode) the corpus, then run every v3 variant: grid + summary."""
    global NORM, FOLD_MODE, RUNS, METRICS
    start_time = time.time()
    tmp = Path(tempfile.mkdtemp())
    cache = OUT / "corpus_grid_v1.npz"
    found = find_cached_corpus()
    if found is not None and not cache.exists():
        print("re-using feature cache:", found)
        shutil.copy(found, cache)
    if cache.exists():
        corpus_raw = np.load(cache, allow_pickle=True)
    else:
        dataroot = find_data_root(start_time)
        corpus_raw = build_corpus(dataroot, tmp / "work", cache,
                                  fast_max=6 if FAST else None)

    configs = build_configs()
    variants = [v.strip().split(":") for v in VARIANTS.split(",") if v.strip()]
    for norm, fold_mode in variants:
        assert norm in ("none", "fold", "subject"), norm
        assert fold_mode in ("session", "person_strat"), fold_mode
        NORM, FOLD_MODE = norm, fold_mode
        RUNS = OUT / f"gtao_grid_v3_{norm}_{fold_mode}"
        RUNS.mkdir(parents=True, exist_ok=True)
        METRICS = OUT / f"grid_v3_{norm}_{fold_mode}_metrics.json"
        corpus = subject_standardise(corpus_raw) if norm == "subject" else corpus_raw
        print(f"\n##### variant norm={norm} folds={fold_mode} -> {RUNS.name}", flush=True)

        ys = corpus["y"]
        keep = np.ones(len(ys), dtype=bool)
        for stream in STREAMS:
            keep &= corpus[f"avail_{stream}"]
        log_step("corpus", n_windows=len(ys),
                 n_subjects=len(set(np.asarray(corpus["subjects"]).tolist())),
                 n_persons=len(set(person_of(s) for s in np.asarray(corpus["subjects"]).tolist())),
                 class_dist=[int(c) for c in np.bincount(ys, minlength=N_CLASSES)],
                 complete_windows=int(keep.sum()),
                 norm=norm, folds=fold_mode)
        for seed in SEEDS:
            run_grid(corpus, seed, configs)
        summarize(SEEDS, configs, write_to=METRICS)
        doc = json.loads(METRICS.read_text())
        doc["status"] = "done"
        doc["variant"] = {"norm": norm, "folds": fold_mode}
        doc["finished_utc"] = time.strftime("%Y-%m-%d %H:%M:%S")
        doc["wall_s"] = round(time.time() - start_time, 1)
        METRICS.write_text(json.dumps(doc, indent=2))

    if RUN_H4:
        h4 = run_masked_eval(corpus_raw, seeds=SEEDS)
        (OUT / "h4_masked_eval.json").write_text(json.dumps(h4, indent=2))
    print("total wall:", round(time.time() - start_time, 1), "s")


if __name__ == "__main__":
    main()
