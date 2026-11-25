"""Build the Phase 10B kernel (learned face encoder): analysis_v3/phase10a.py + phase10b.py in one file.

    python make_kernel.py && kaggle kernels push -p kernel
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ANALYSIS = HERE.parent.parent / "analysis_v3"
folder = HERE / "kernel"
folder.mkdir(exist_ok=True)
a = (ANALYSIS / "phase10a.py").read_text(encoding="utf-8")
a = a[:a.index('if __name__ == "__main__":')]
b = (ANALYSIS / "phase10b.py").read_text(encoding="utf-8")
b = "\n".join(line for line in b.splitlines() if not line.endswith("# kernel: inlined")) + "\n"
(folder / "phase10b.py").write_text("# ---- phase10a.py (verbatim, without its main call) ----\n" + a
                                    + "\n# ---- phase10b.py ----\n" + b, encoding="utf-8")
meta = {"id": "zakariamakhas/gtao-v12-dipser-face-encoder", "title": "gTAO v12 DIPSER face encoder",
        "code_file": "phase10b.py", "language": "python", "kernel_type": "script",
        "is_private": True, "enable_gpu": False, "enable_tpu": False, "enable_internet": False,
        "dataset_sources": ["zakariamakhas/dipser-full-corpus-features",
                            "zakariamakhas/dipser-face-embeddings"],
        "competition_sources": [], "model_sources": [], "kernel_sources": []}
(folder / "kernel-metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
print("wrote kernel/")
