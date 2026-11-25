"""Build the Phase 10A kernel (sparse-probe tracking): analysis_v3/phase10a.py, run as is.

    python make_kernel.py && kaggle kernels push -p kernel
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
folder = HERE / "kernel"
folder.mkdir(exist_ok=True)
(folder / "phase10a.py").write_text((HERE.parent.parent / "analysis_v3" / "phase10a.py").read_text(encoding="utf-8"), encoding="utf-8")
meta = {"id": "zakariamakhas/gtao-v11-dipser-sparse-probe", "title": "gTAO v11 DIPSER sparse probe",
        "code_file": "phase10a.py", "language": "python", "kernel_type": "script",
        "is_private": True, "enable_gpu": False, "enable_tpu": False, "enable_internet": False,
        "dataset_sources": ["zakariamakhas/dipser-full-corpus-features"], "competition_sources": [],
        "model_sources": [], "kernel_sources": []}
(folder / "kernel-metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
print("wrote kernel/")
