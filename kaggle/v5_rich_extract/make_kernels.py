"""Build the Phase 6 Kaggle kernels from driver.py + the PC pipeline sources (verbatim).

    python make_kernels.py --shards 5        # writes test/ and shard0..4/
    kaggle kernels push -p test               # first: speed + frames + self-test
"""
import argparse
import json
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
FILES = ["pc_pipeline/extract_features.py", "pc_pipeline/_common.py",
         "kaggle/v2_nested/dipser_grid/gtao_grid.py"]


def build(mode, shard, n_shards, seq_k, face_k):
    src = (HERE / "driver.py").read_text(encoding="utf-8")
    embedded = {f: (REPO / f).read_text(encoding="utf-8") for f in FILES}
    replacements = {
        'MODE = "test"            # set by make_kernels.py': f'MODE = "{mode}"',
        "SHARD, N_SHARDS = 0, 1   # set by make_kernels.py": f"SHARD, N_SHARDS = {shard}, {n_shards}",
        "SEQ_K, FACE_K = 12, 24   # set by make_kernels.py": f"SEQ_K, FACE_K = {seq_k}, {face_k}",
        "EMBEDDED = {}            # filled by make_kernels.py: relative path -> source text":
            "EMBEDDED = " + repr(embedded),
    }
    if os.environ.get("SCIDB_USER") and os.environ.get("SCIDB_PASS"):
        # user decision 2026-11-19: temporary read-only account (expires 2026-11-24) embedded
        # in the PRIVATE kernels, because API-pushed runs cannot read Kaggle Secrets.
        replacements["EMBEDDED_USER, EMBEDDED_PASS = None, None   # set by make_kernels.py from env (never in git)"] = \
            f"EMBEDDED_USER, EMBEDDED_PASS = {os.environ['SCIDB_USER']!r}, {os.environ['SCIDB_PASS']!r}"
    for old, new in replacements.items():
        assert src.count(old) == 1, old
        src = src.replace(old, new)
    name = "test" if mode == "test" else f"shard{shard}"
    folder = HERE / name
    folder.mkdir(exist_ok=True)
    (folder / "rich_extract.py").write_text(src, encoding="utf-8")
    meta = {"id": f"zakariamakhas/gtao-v5-dipser-rich-extract-{name}",
            "title": f"gTAO v5 DIPSER rich extract {name}",
            "code_file": "rich_extract.py", "language": "python", "kernel_type": "script",
            "is_private": True, "enable_gpu": False, "enable_tpu": False,
            "enable_internet": True,
            "dataset_sources": ["zakariamakhas/dipser-full-corpus-features"],
            "competition_sources": [], "kernel_sources": [], "model_sources": []}
    (folder / "kernel-metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
    print("wrote", name)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", type=int, default=5)
    ap.add_argument("--seq-k", type=int, default=12)
    ap.add_argument("--face-k", type=int, default=24)
    a = ap.parse_args()
    build("test", 0, 1, a.seq_k, a.face_k)
    for k in range(a.shards):
        build("shard", k, a.shards, a.seq_k, a.face_k)
