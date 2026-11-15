"""Measure the computational cost of gTAO-Net on a local CPU-only machine.

Run from the repository root (needs torch, numpy, mpmath, pillow, scikit-learn):

    python verify/local_cost_benchmark.py --kernel <path to gtao_grid.py>

It imports the DIPSER Kaggle kernel itself (gtao_grid.py, the v2 nested version)
so the model, the training loop (fit_fold) and the Tchebichef transforms are
exactly those of the paper. The data are SYNTHETIC numbers with the real DIPSER
shapes (1,566 windows, 23 session-subjects, 3 classes, feature sizes
1,536 / 8,604 / 18 / 48): cost depends on the shapes, not on the values.
Scores are meaningless here; only time, memory and parameter counts are reported.
"""
import argparse
import importlib.util
import json
import platform
import resource
import sys
import time
from pathlib import Path

import numpy as np

N_WINDOWS = 1566
N_SUBJECTS = 23
FRAMES_PER_WINDOW = 40          # 4-s window at about 10 frames per second
N_FOLDS = 5


def load_kernel(path):
    spec = importlib.util.spec_from_file_location("gtao_grid", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def peak_memory_mb():
    """Peak resident memory of this process, in MB (macOS reports bytes, Linux KB)."""
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if platform.system() == "Darwin":
        return peak / (1024 * 1024)
    return peak / 1024


def count_parameters(model):
    total = 0
    for parameter in model.parameters():
        total = total + parameter.numel()
    return total


def time_feature_encoding(k, n_windows):
    """Tchebichef encoding of n_windows synthetic windows (face 2-D+1-D, mesh, pose, physio)."""
    rng = np.random.default_rng(0)
    start = time.perf_counter()
    for _ in range(n_windows):
        faces = []
        for _ in range(FRAMES_PER_WINDOW):
            pixels = rng.random((k.FACE_SIZE, k.FACE_SIZE), dtype=np.float32)
            faces.append(k.transform_2d(pixels, k.FACE_K))
        face = k.sequence_moments(k.pad_time(np.stack(faces), k.SEQ_K), -1)
        mesh = k.sequence_moments(k.pad_time(rng.random((FRAMES_PER_WINDOW, 478, 3), dtype=np.float32), k.SEQ_K), 478 * 3)
        pose = k.sequence_moments(k.pad_time(rng.random((FRAMES_PER_WINDOW, 3), dtype=np.float32), k.SEQ_K), 3)
        physio = k.transform_1d(rng.random((8, FRAMES_PER_WINDOW), dtype=np.float32), k.SEQ_K).ravel()
    seconds = time.perf_counter() - start
    return seconds, [len(face), len(mesh), len(pose), len(physio)]


def synthetic_corpus(dims):
    rng = np.random.default_rng(1)
    features = {}
    for stream in dims:
        features[stream] = rng.standard_normal((N_WINDOWS, dims[stream])).astype(np.float32)
    labels = rng.choice(3, size=N_WINDOWS, p=[0.43, 0.56, 0.01]).astype(np.int64)
    subjects = np.array(["s%02d" % (i % N_SUBJECTS) for i in range(N_WINDOWS)])
    return features, labels, subjects


def run_config(k, torch, dims, features, labels, subjects, fusion, use_skip, unimodal):
    """Train the 5 outer folds of one configuration (one seed). Returns per-fold seconds and epochs."""
    stream_list = sorted(dims)
    if unimodal:
        stream_list = [unimodal]
    fold_of = {}
    unique = sorted(set(subjects.tolist()))
    for position, subject in enumerate(unique):
        fold_of[subject] = position % N_FOLDS
    fold_array = np.array([fold_of[s] for s in subjects])
    seconds = []
    epochs = []
    n_params = None
    for fold in range(N_FOLDS):
        train_rows = np.flatnonzero(fold_array != fold)
        test_rows = np.flatnonzero(fold_array == fold)
        k.set_seed(int("0" + str(fold)))
        model_dims = {}
        for stream in stream_list:
            model_dims[stream] = dims[stream]
        if unimodal:
            model = k.GTAONetGrid(model_dims, fusion="core", use_skip=False, mdrop=False)
        else:
            model = k.GTAONetGrid(model_dims, fusion=fusion, use_skip=use_skip, mdrop=False)
        n_params = count_parameters(model)
        train_features = {}
        test_features = {}
        for stream in stream_list:
            train_features[stream] = torch.as_tensor(features[stream][train_rows])
            test_features[stream] = torch.as_tensor(features[stream][test_rows])
        start = time.perf_counter()
        best, trace, metas, inner = k.fit_fold(model, train_features, torch.as_tensor(labels[train_rows]),
                                               test_features, torch.as_tensor(labels[test_rows]),
                                               stream_list=stream_list, loss="coral",
                                               subj_tr=subjects[train_rows])
        seconds.append(time.perf_counter() - start)
        epochs.append(int(metas["epochs"]))
    return seconds, epochs, n_params


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kernel", required=True, help="path to the v2 DIPSER kernel gtao_grid.py")
    parser.add_argument("--out", default="verify/local_cost_benchmark.json")
    args = parser.parse_args()

    k = load_kernel(args.kernel)
    import torch
    torch.set_num_threads(max(1, torch.get_num_threads()))
    k.DEVICE = "cpu" if hasattr(k, "DEVICE") else None

    report = {"machine": {"platform": platform.platform(), "processor": platform.processor(),
                          "python": sys.version.split()[0], "torch": torch.__version__,
                          "torch_threads": torch.get_num_threads(), "gpu": "none (CPU only)"}}

    enc_seconds, enc_dims = time_feature_encoding(k, 200)
    report["tchebichef_encoding"] = {
        "windows_timed": 200, "frames_per_window": FRAMES_PER_WINDOW,
        "ms_per_window": round(1000 * enc_seconds / 200, 2),
        "estimated_s_for_1566_windows": round(enc_seconds / 200 * N_WINDOWS, 1),
        "feature_dims": enc_dims,
        "note": "moments only; image decoding and the shipped MediaPipe landmarks are not included"}

    dims = {"face": 1536, "facemesh": 8604, "head_pose": 18, "physio": 48}
    features, labels, subjects = synthetic_corpus(dims)
    bytes_total = 0
    for stream in dims:
        bytes_total = bytes_total + features[stream].nbytes
    report["feature_cache_mb_float32"] = round(bytes_total / (1024 * 1024), 1)

    configs = [("largest fusion: gate + link, 4 streams", "gate", True, None),
               ("core fusion, no link, 4 streams", "core", False, None),
               ("single stream: face mesh", None, False, "facemesh")]
    report["training"] = []
    for label, fusion, use_skip, unimodal in configs:
        seconds, epochs, n_params = run_config(k, torch, dims, features, labels, subjects, fusion, use_skip, unimodal)
        per_epoch = []
        for s, e in zip(seconds, epochs):
            per_epoch.append(s / e)
        report["training"].append({
            "config": label, "parameters": n_params,
            "seconds_per_fold": [round(s, 2) for s in seconds],
            "epochs_per_fold": epochs,
            "mean_seconds_per_fold": round(float(np.mean(seconds)), 2),
            "mean_seconds_per_epoch": round(float(np.mean(per_epoch)), 3),
            "upper_bound_s_per_fold_40_epochs": round(float(np.mean(per_epoch)) * 40, 1)})
        print(label, report["training"][-1])
    report["peak_memory_mb"] = round(peak_memory_mb(), 0)
    Path(args.out).write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
