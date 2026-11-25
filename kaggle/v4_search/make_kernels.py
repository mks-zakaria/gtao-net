"""One Kaggle CPU kernel per (dataset, objective) from search.py.

    python make_kernels.py && for d in */; do kaggle kernels push -p "$d"; done
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
SOURCE = (HERE / "search.py").read_text(encoding="utf-8")
SOURCES = {"dipser": {"dataset_sources": ["zakariamakhas/dipser-full-corpus-features"],
                      "kernel_sources": []},
           "dipser_rich": {"dataset_sources": [],
                           "kernel_sources": ["zakariamakhas/gtao-v5-dipser-rich-merge"]},
           "dipser_crop": {"dataset_sources": [],
                           "kernel_sources": ["zakariamakhas/gtao-v5-dipser-rich-merge",
                                              "zakariamakhas/gtao-v6-dipser-face-crop-merge"]},
           "daisee": {"dataset_sources": [],
                      "kernel_sources": ["zakariamakhas/gtao-net-daisee-v2-nested-early-stopping"]}}
import sys
DATASETS = sys.argv[1:] or ["dipser", "daisee"]
for dataset in DATASETS:
    for objective in ("acc", "qwk"):
        code = (SOURCE
                .replace('os.environ.get("V4_DATASET", "dipser")', f'os.environ.get("V4_DATASET", "{dataset}")')
                .replace('os.environ.get("V4_OBJECTIVE", "acc")', f'os.environ.get("V4_OBJECTIVE", "{objective}")'))
        assert f'"{dataset}")' in code and f'"{objective}")' in code
        folder = HERE / f"{dataset}_{objective}"
        folder.mkdir(exist_ok=True)
        (folder / "search.py").write_text(code, encoding="utf-8")
        meta = {"id": f"zakariamakhas/gtao-v4-bayes-search-{dataset.replace('_', '-')}-{objective}",
                "title": f"gTAO v4 Bayes search {dataset} {objective}",
                "code_file": "search.py", "language": "python", "kernel_type": "script",
                "is_private": True, "enable_gpu": False, "enable_tpu": False,
                "enable_internet": False, "competition_sources": [], "model_sources": [],
                **SOURCES[dataset]}
        (folder / "kernel-metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
        print("wrote", folder.name)
