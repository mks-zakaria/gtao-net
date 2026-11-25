"""Build the Phase 9 kernel: class_context.py with kaggle/v8_context/context.py embedded verbatim.

    python make_kernel.py && kaggle kernels push -p kernel
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
src = (HERE / "class_context.py").read_text(encoding="utf-8")
ctx = (HERE.parent / "v8_context" / "context.py").read_text(encoding="utf-8")
old = "CONTEXT_SRC = None   # make_kernel.py: source of kaggle/v8_context/context.py"
assert src.count(old) == 1
src = src.replace(old, "CONTEXT_SRC = " + repr(ctx))
folder = HERE / "kernel"
folder.mkdir(exist_ok=True)
(folder / "class_context.py").write_text(src, encoding="utf-8")
meta = {"id": "zakariamakhas/gtao-v10-dipser-class-context", "title": "gTAO v10 DIPSER class context",
        "code_file": "class_context.py", "language": "python", "kernel_type": "script",
        "is_private": True, "enable_gpu": False, "enable_tpu": False, "enable_internet": False,
        "dataset_sources": [], "competition_sources": [], "model_sources": [],
        "kernel_sources": ["zakariamakhas/gtao-v5-dipser-rich-merge", "zakariamakhas/gtao-v6-dipser-face-crop-merge"]}
(folder / "kernel-metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
print("wrote kernel/")
