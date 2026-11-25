"""Phase 7 Kaggle kernels: face-crop moment families, downloaded from SciDB FTP.

Reuses kaggle/v5_rich_extract/driver.py (FTP walk, resumable download, shard loop) with
three textual changes (each asserted): the extractor module (extract_families instead of
extract_features), the output folder, and no paper self-test inside the shards (the merge
kernel tests the full-frame face against corpus_rich instead). Sources embedded verbatim.

    SCIDB_USER=... SCIDB_PASS=... python make_kernels.py     # writes shard0..4/ (gitignored)
"""
import json
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
FILES = ["pc_pipeline/extract_families.py", "pc_pipeline/moment_families.py",
         "pc_pipeline/_common.py", "kaggle/v2_nested/dipser_grid/gtao_grid.py"]
N_SHARDS, SEQ_K, FACE_K = 5, 8, 16


def build(shard):
    src = (REPO / "kaggle/v5_rich_extract/driver.py").read_text(encoding="utf-8")
    embedded = {f: (REPO / f).read_text(encoding="utf-8") for f in FILES}
    replacements = {
        'MODE = "test"            # set by make_kernels.py': 'MODE = "shard"',
        "SHARD, N_SHARDS = 0, 1   # set by make_kernels.py": f"SHARD, N_SHARDS = {shard}, {N_SHARDS}",
        "SEQ_K, FACE_K = 12, 24   # set by make_kernels.py": f"SEQ_K, FACE_K = {SEQ_K}, {FACE_K}",
        "EMBEDDED = {}            # filled by make_kernels.py: relative path -> source text":
            "EMBEDDED = " + repr(embedded),
        "    import extract_features as E\n": "    import extract_families as E\n",
        'RICH = OUT / "rich"': 'RICH = OUT / "crop"',
        "        corpus = find_corpus()\n": "        corpus = None   # v6: identity test runs in the merge kernel\n",
    }
    user, password = os.environ.get("SCIDB_USER"), os.environ.get("SCIDB_PASS")
    if not (user and password):
        raise SystemExit("set SCIDB_USER / SCIDB_PASS (never committed)")
    replacements["EMBEDDED_USER, EMBEDDED_PASS = None, None   # set by make_kernels.py from env (never in git)"] = \
        f"EMBEDDED_USER, EMBEDDED_PASS = {user!r}, {password!r}"
    for old, new in replacements.items():
        assert src.count(old) == 1, old
        src = src.replace(old, new)
    folder = HERE / f"shard{shard}"
    folder.mkdir(exist_ok=True)
    (folder / "crop_extract.py").write_text(src, encoding="utf-8")
    meta = {"id": f"zakariamakhas/gtao-v6-dipser-face-crop-shard{shard}",
            "title": f"gTAO v6 DIPSER face crop shard{shard}",
            "code_file": "crop_extract.py", "language": "python", "kernel_type": "script",
            "is_private": True, "enable_gpu": False, "enable_tpu": False, "enable_internet": True,
            "dataset_sources": [], "competition_sources": [], "kernel_sources": [], "model_sources": []}
    (folder / "kernel-metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
    print("wrote", folder.name)


if __name__ == "__main__":
    for k in range(N_SHARDS):
        build(k)
