"""Statistics on Kaggle: every committed analysis script, run on Kaggle outputs.

The scripts (analysis_v3/*.py, kaggle/v4_search/search.py) are embedded verbatim by
make_kernel.py (sha256 printed). LINKS lays the attached kernel outputs / dataset files out
in the repo layout the scripts expect (file-by-file symlinks; several kernels may fill one
folder, e.g. the 3 v3b seeds -> results/v3b/merged). STEPS are the original command lines,
each run as a subprocess from its original folder; stdout goes to results/logs/<name>.txt.
Every file a step writes (not an input link) is copied to /kaggle/working/results.
"""
import hashlib
import os
import shutil
import subprocess
import sys
from pathlib import Path

EMBEDDED = {}   # relative path -> source text (make_kernel.py)
LINKS = []      # (kernel slug or "dataset:<file>", target folder or file, relative to ROOT[, "dirs"])
STEPS = []      # (name, cwd, argv after the interpreter, extra env)
ON_KAGGLE = os.path.isdir("/kaggle/working")
ROOT = Path("/kaggle/working/code") if ON_KAGGLE else Path("v9_code").resolve()
INPUT = Path(os.environ.get("V9_INPUT", "/kaggle/input"))
SKIP = {"__output__.json", "__results__.html", "__notebook__.ipynb", "custom.css"}

for rel, text in EMBEDDED.items():
    target = ROOT / rel
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding="utf-8")
    print(f"code {rel}: sha256 {hashlib.sha256(text.encode('utf-8')).hexdigest()[:16]}", flush=True)


def slug_root(slug):
    for folder, dirs, _ in os.walk(INPUT):
        if Path(folder).name == slug:
            return Path(folder)
    return None


def find_file(name):
    for folder, _, names in os.walk(INPUT):
        if name in names:
            return Path(folder) / name
    return None


linked = set()
for source, target, *mode in LINKS:
    if source.startswith("dataset:"):
        path = find_file(source.split(":", 1)[1])
        assert path is not None, f"missing dataset file {source}"
        dest = ROOT / target
        dest.parent.mkdir(parents=True, exist_ok=True)
        os.symlink(path, dest)
        linked.add(dest)
        print(f"input {source} -> {target}", flush=True)
        continue
    root = slug_root(source)
    assert root is not None, f"missing kernel output {source}"
    n = 0
    for folder, _, names in os.walk(root):
        for name in names:
            if name in SKIP or name.endswith(".log"):
                continue
            if mode == ["dirs"] and Path(folder) == root:   # merge: run folders only
                continue
            src = Path(folder) / name
            dest = ROOT / target / src.relative_to(root)
            if dest.exists() or dest.is_symlink():
                raise SystemExit(f"two inputs for one path: {dest}")
            dest.parent.mkdir(parents=True, exist_ok=True)
            os.symlink(src, dest)
            linked.add(dest)
            n += 1
    print(f"input {source} -> {target}: {n} files", flush=True)

for d in ("results/v4", "results/v6", "results/v8", "results/v10"):   # output folders of the steps
    (ROOT / d).mkdir(parents=True, exist_ok=True)
logs = ROOT / "results" / "logs"
logs.mkdir(parents=True, exist_ok=True)
failed = []
for name, cwd, argv, env in STEPS:
    print(f"\n===== {name}: (cd {cwd}) python {' '.join(argv)}", flush=True)
    env = {k: v.replace("__ROOT__", str(ROOT)) for k, v in env.items()}
    run = subprocess.run([sys.executable] + argv, cwd=ROOT / cwd, env={**os.environ, **env},
                         capture_output=True, text=True)
    (logs / f"{name}.txt").write_text(run.stdout)
    print(run.stdout[-4000:], flush=True)
    if run.returncode != 0:
        print(run.stderr[-4000:], flush=True)
        failed.append(name)

out = Path("/kaggle/working/results") if ON_KAGGLE else Path("v9_results").resolve()
for base in ("results", "paper"):
    for p in (ROOT / base).rglob("*"):
        if p.is_file() and not p.is_symlink() and p not in linked:
            dest = out / p.relative_to(ROOT)
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(p, dest)
            print("wrote", dest.relative_to(out))
if failed:
    raise SystemExit(f"steps failed: {failed}")
