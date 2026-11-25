"""Build the Kaggle statistics kernel from driver.py + the committed analysis scripts.

    python make_kernel.py && kaggle kernels push -p kernel

Every step is the command line that produced the committed result (see STATUS.md).
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
CODE = ["analysis_v3/whowhen.py", "analysis_v3/label_prior.py", "analysis_v3/split_leakage.py",
        "analysis_v3/compare_v3.py", "analysis_v3/phase1b.py", "analysis_v3/gate3.py",
        "analysis_v3/rich_compare.py", "analysis_v3/phase7.py", "kaggle/v4_search/search.py"]
CORPUS = "data/corpus_full.npz"
LINKS = [("dataset:corpus_full.npz", CORPUS),
         ("gtao-net-dipser-grid-v2-nested-early-stopping", "results/v2/raw_dipser"),
         ("gtao-net-daisee-v2-nested-early-stopping", "results/v2/raw_daisee"),
         ("gtao-net-engagenet-v2-nested-early-stopping", "results/v2/raw_engagenet"),
         ("gtao-net-dipser-grid-v3-standardised", "results/v3")]
LINKS += [(f"gtao-net-dipser-v3b-full-corpus-seed{k}", "results/v3b/merged", "dirs") for k in range(3)]
LINKS += [(f"gtao-v4-bayes-search-{d.replace('_', '-')}-{o}", f"results/v4/{d}_{o}")
          for d in ("dipser", "dipser_rich", "dipser_crop") for o in ("acc", "qwk")]
STEPS = [
    ("label_prior", "analysis_v3", ["label_prior.py", f"../{CORPUS}"], {}),
    ("compare_v3", ".", ["analysis_v3/compare_v3.py", "--gate", "--json", "results/v3/report.json",
                        "--run", "v2=results/v2/raw_dipser/gtao_grid_v1",
                        "--run", "v3 none=results/v3/gtao_grid_v3_none_person_strat",
                        "--run", "v3 fold=results/v3/gtao_grid_v3_fold_person_strat",
                        "--run", "v3 subject=results/v3/gtao_grid_v3_subject_person_strat"], {}),
    ("phase1b", "analysis_v3", ["phase1b.py", "--corpus", f"../{CORPUS}"], {}),
    ("gate3", "analysis_v3", ["gate3.py", "--out", "../results/v3b/merged", "--boot", "2000",
                              "--json", "../results/v3b/gate3.json"], {}),
    ("split_leakage", ".", ["analysis_v3/split_leakage.py"],
     {"V4_MODELS": "mlp", "V4_CORPUS": f"__ROOT__/{CORPUS}", "V4_OUT": "__ROOT__/v4_out"}),
    ("rich_compare", "analysis_v3", ["rich_compare.py"], {}),
    ("phase7", "analysis_v3", ["phase7.py"], {}),
]
if (REPO / "analysis_v3/phase8.py").exists():           # time context, once analysed
    CODE.append("analysis_v3/phase8.py")
    LINKS.append(("gtao-v8-dipser-time-context", "results/v8"))
    STEPS.append(("phase8", "analysis_v3", ["phase8.py"], {}))
if (REPO / "analysis_v3/phase9.py").exists():           # class context
    CODE.append("analysis_v3/phase9.py")
    LINKS.append(("gtao-v10-dipser-class-context", "results/v10"))
    STEPS.append(("phase9", "analysis_v3", ["phase9.py"], {}))

src = (HERE / "driver.py").read_text(encoding="utf-8")
for old, new in {"EMBEDDED = {}   # relative path -> source text (make_kernel.py)":
                     "EMBEDDED = " + repr({s: (REPO / s).read_text(encoding="utf-8") for s in CODE}),
                 'LINKS = []      # (kernel slug or "dataset:<file>", target folder or file, relative to ROOT[, "dirs"])':
                     "LINKS = " + repr(LINKS),
                 "STEPS = []      # (name, cwd, argv after the interpreter, extra env)":
                     "STEPS = " + repr(STEPS)}.items():
    assert src.count(old) == 1, old
    src = src.replace(old, new)
folder = HERE / "kernel"
folder.mkdir(exist_ok=True)
(folder / "analysis.py").write_text(src, encoding="utf-8")
kernels = sorted({link[0] for link in LINKS if not link[0].startswith("dataset:")})
meta = {"id": "zakariamakhas/gtao-v9-statistics", "title": "gTAO v9 statistics",
        "code_file": "analysis.py", "language": "python", "kernel_type": "script",
        "is_private": True, "enable_gpu": False, "enable_tpu": False, "enable_internet": False,
        "dataset_sources": ["zakariamakhas/dipser-full-corpus-features"],
        "competition_sources": [], "model_sources": [],
        "kernel_sources": ["zakariamakhas/" + s for s in kernels]}
(folder / "kernel-metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
print("steps", [s[0] for s in STEPS], "\nkernels", len(kernels))
