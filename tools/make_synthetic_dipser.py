"""Build a tiny synthetic DIPSER-like dataset for CPU smoke tests of gtao_grid.py.

Why this exists
---------------
The real DIPSER slice is ~1 GB per subject and lives on Kaggle. To check that a
code change does not move any number, we only need *some* data that goes through
exactly the same loader code paths. This script writes a few small fake subject
bundles in the same format the loader in ``gtao_grid.py`` expects.

What it writes
--------------
One ``<group>__<experiment>__<subject>.zipbin`` file per subject (a plain ZIP
archive with a different extension). Inside each archive::

    subject_XX/
        images/<HH-MM-SS-ffffff>.png        one grey 64x64 face frame per second
        metadata/<same stem>.json           facemesh (478 xyz points) + head pose
        labels/rater_<r>.json               attention change-points (1..5 scale)
        watch_sensors/watch.json            heart rate, accelerometer, gyroscope, light

Every stream is present in every 4 s window, so all windows are "complete" and
all four streams (face, facemesh, head_pose, physio) are used by the grid.
The labels are loosely tied to image brightness so a model can learn something,
but the exact values do not matter for an equivalence test.

Usage
-----
    python make_synthetic_dipser.py OUTPUT_DIR [--subjects 6] [--seconds 120]

Then point the training script at it::

    DIPSEER_DATA_ROOT=OUTPUT_DIR GRID_FAST=1 python gtao_grid.py
"""
import argparse
import io
import json
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image

N_FACEMESH_POINTS = 478          # MediaPipe facemesh size, the loader hard-codes it
IMAGE_SIZE = 64
START_SECONDS = 10 * 3600        # recordings start at 10:00:00
N_RATERS = 3
SENSOR_RATE_HZ = 2               # watch samples per second (per sensor)


def format_time_stem(seconds_of_day):
    """Turn seconds since midnight into a file stem like '10-00-05-000000'.

    The loader's parse_ts() accepts '-' as a separator and reads the last
    part as microseconds.
    """
    whole_seconds = int(seconds_of_day)
    microseconds = int(round((seconds_of_day - whole_seconds) * 1e6))
    hours = whole_seconds // 3600
    minutes = (whole_seconds % 3600) // 60
    seconds = whole_seconds % 60
    return f"{hours:02d}-{minutes:02d}-{seconds:02d}-{microseconds:06d}"


def format_label_datetime(seconds_of_day):
    """Rater files use 'HH:MM:SS' timestamps."""
    whole_seconds = int(seconds_of_day)
    hours = whole_seconds // 3600
    minutes = (whole_seconds % 3600) // 60
    seconds = whole_seconds % 60
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def make_attention_curve(rng, n_seconds):
    """A slowly changing 'true' attention level (1..5) for every second."""
    levels = []
    current = int(rng.integers(1, 6))
    for _ in range(n_seconds):
        if rng.random() < 0.08:          # occasionally jump to a new level
            current = int(rng.integers(1, 6))
        levels.append(current)
    return levels


def png_bytes(pixel_array):
    """Encode a uint8 2-D array as PNG bytes."""
    buffer = io.BytesIO()
    Image.fromarray(pixel_array, mode="L").save(buffer, format="PNG")
    return buffer.getvalue()


def build_subject_zip(zip_path, subject_name, rng, n_seconds):
    """Write one synthetic subject bundle to ``zip_path``."""
    attention = make_attention_curve(rng, n_seconds)
    base_mesh = rng.normal(0.0, 1.0, size=(N_FACEMESH_POINTS, 3))

    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        # Face frames + per-frame metadata (one per second).
        for second in range(n_seconds):
            t = START_SECONDS + second
            stem = format_time_stem(t)
            brightness = 40 + 40 * attention[second]
            noise = rng.normal(0.0, 20.0, size=(IMAGE_SIZE, IMAGE_SIZE))
            pixels = np.clip(brightness + noise, 0, 255).astype(np.uint8)
            archive.writestr(f"{subject_name}/images/{stem}.png", png_bytes(pixels))

            mesh = base_mesh + rng.normal(0.0, 0.05, size=base_mesh.shape)
            mesh = mesh + 0.1 * attention[second]
            mesh_points = []
            for point in mesh:
                mesh_points.append({"x": float(point[0]), "y": float(point[1]),
                                    "z": float(point[2])})
            metadata = {
                "person": {"face": {"facemesh": mesh_points}},
                "head_pose": {"pitch": float(rng.normal(attention[second], 1.0)),
                              "yaw": float(rng.normal(0.0, 5.0)),
                              "roll": float(rng.normal(0.0, 5.0))},
            }
            archive.writestr(f"{subject_name}/metadata/{stem}.json", json.dumps(metadata))

        # Raters: noisy copies of the attention curve, as change-points.
        for rater in range(N_RATERS):
            change_points = []
            previous = None
            for second in range(n_seconds):
                value = attention[second]
                if rng.random() < 0.15:
                    value = int(np.clip(value + rng.integers(-1, 2), 1, 5))
                if value != previous:
                    change_points.append({
                        "datetime": format_label_datetime(START_SECONDS + second),
                        "attention": value})
                    previous = value
            archive.writestr(f"{subject_name}/labels/rater_{rater}.json",
                             json.dumps(change_points))

        # Watch sensors in the "data" schema: {"data": {sensor: [records]}}.
        sensors = {"heart_rate": [], "accelerometer": [], "gyroscope": [], "light": []}
        n_samples = n_seconds * SENSOR_RATE_HZ
        for sample in range(n_samples):
            t = START_SECONDS + sample / SENSOR_RATE_HZ
            level = attention[min(int(sample / SENSOR_RATE_HZ), n_seconds - 1)]
            sensors["heart_rate"].append(
                {"timestamp": t, "value0": float(rng.normal(70 + 3 * level, 2.0))})
            accel = rng.normal(0.0, 1.0, size=3)
            sensors["accelerometer"].append(
                {"timestamp": t, "value0": float(accel[0]), "value1": float(accel[1]),
                 "value2": float(accel[2])})
            gyro = rng.normal(0.0, 0.5, size=3)
            sensors["gyroscope"].append(
                {"timestamp": t, "value0": float(gyro[0]), "value1": float(gyro[1]),
                 "value2": float(gyro[2])})
            sensors["light"].append(
                {"timestamp": t, "value0": float(rng.normal(300 + 10 * level, 5.0))})
        archive.writestr(f"{subject_name}/watch_sensors/watch.json",
                         json.dumps({"data": sensors}))


def main():
    """Parse the command line and write all subject bundles."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("output_dir")
    parser.add_argument("--subjects", type=int, default=6)
    parser.add_argument("--seconds", type=int, default=120)
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    for index in range(args.subjects):
        group = f"group_{index // 3 + 1:02d}"
        subject = f"subject_{index + 1:02d}"
        bundle_name = f"{group}__experiment_01__{subject}.zipbin"
        build_subject_zip(output_dir / bundle_name, subject, rng, args.seconds)
        print("wrote", output_dir / bundle_name)


if __name__ == "__main__":
    main()
