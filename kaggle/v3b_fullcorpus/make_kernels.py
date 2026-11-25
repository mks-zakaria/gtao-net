"""Write one Kaggle kernel folder per seed from dipser_grid/gtao_grid.py.

The kernels differ only in the default of GTAO_SEEDS. Re-run after editing the script:
    python make_kernels.py && kaggle kernels push -p seed0   (seed1, seed2)
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
SOURCE = (HERE / "dipser_grid" / "gtao_grid.py").read_text(encoding="utf-8")
DEFAULT = 'os.environ.get("GTAO_SEEDS", "0,1,2")'
assert SOURCE.count(DEFAULT) == 1

for seed in (0, 1, 2):
    folder = HERE / f"seed{seed}"
    folder.mkdir(exist_ok=True)
    (folder / "gtao_grid.py").write_text(
        SOURCE.replace(DEFAULT, f'os.environ.get("GTAO_SEEDS", "{seed}")'), encoding="utf-8")
    meta = {
        "id": f"zakariamakhas/gtao-net-dipser-v3b-full-corpus-seed{seed}",
        "title": f"gTAO-Net DIPSER v3b full corpus seed{seed}",
        "code_file": "gtao_grid.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": False,            # CPU: no GPU quota (Amendment 2)
        "enable_tpu": False,
        "enable_internet": False,
        "dataset_sources": ["zakariamakhas/dipser-full-corpus-features"],
        "competition_sources": [],
        "kernel_sources": [],
        "model_sources": [],
    }
    (folder / "kernel-metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
    print("wrote", folder)
