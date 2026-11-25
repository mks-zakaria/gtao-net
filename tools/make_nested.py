"""Derive the v2 (nested early-stopping) Kaggle scripts from the v1 originals.

v1 bug: fit_fold() early-stopped on the SAME held-out fold that is then scored,
so every pooled QWK in v1 is optimistically biased. v2 holds out INNER_FRAC of
the *training-fold* subjects as an inner validation set for epoch selection;
the outer fold is used only for the final prediction.

Usage: python tools/make_nested.py   (from the bundle root). Idempotent.

RETIRED 2026-11-04. Kept only as a record of the first v2 patch. It is wrong for DAiSEE and
EngageNet: their fit_fold has no docstring, so the guard landed in official_split_folds()
and nested early stopping never ran. The scripts in kaggle/v2_nested/ were then rewritten
by hand for readability, with the guard in fit_fold and both inner parts required to hold
two classes. Running this file would overwrite them, so it now refuses unless --force.
"""
import re
import sys
from pathlib import Path

if "--force" not in sys.argv:
    sys.exit("make_nested.py is retired (see its docstring); kaggle/v2_nested/ is now the source.")

ROOT = Path(__file__).resolve().parents[1]
JOBS = [("dipser_grid", "gtao_grid.py"),
        ("daisee", "daisee-gtao.py"),
        ("engagenet", "engagenet-gtao-scoped.py")]

CONST = '''PATIENCE = 8
# v2: nested early stopping. Epoch selection uses an inner validation split of
# the TRAINING-fold subjects; the scored outer fold never influences training.
NESTED = os.environ.get("GTAO_NESTED", "1") == "1"
INNER_FRAC = 0.2
'''

HELPER = '''def _inner_split(Xtr, ytr, softtr, subj_tr):
    """Split the training fold by subject: (1-INNER_FRAC) train, INNER_FRAC val."""
    subj_tr = np.asarray(subj_tr).astype(str)
    uniq = np.array(sorted(set(subj_tr.tolist())))
    n_val = max(1, int(round(INNER_FRAC * len(uniq))))
    for _ in range(20):                      # re-draw until both sides are usable
        val_s = set(uniq[np.random.permutation(len(uniq))[:n_val]].tolist())
        m_np = np.array([s in val_s for s in subj_tr])
        if (~m_np).any() and m_np.any() and len(np.unique(ytr.cpu().numpy()[~m_np])) >= 2:
            break
    else:
        raise RuntimeError("could not draw a usable inner split")
    m = torch.as_tensor(m_np, device=ytr.device)
    pick = lambda X, mask: {k: v[mask] for k, v in X.items()}
    return (pick(Xtr, ~m), ytr[~m], (softtr[~m] if softtr is not None else None),
            pick(Xtr, m), ytr[m])


'''

GUARD = '''    if NESTED:
        if subj_tr is None:
            raise ValueError("NESTED=1 requires subj_tr (training-fold subject ids)")
        Xtr, ytr, softtr, Xva, yva = _inner_split(Xtr, ytr, softtr, subj_tr)
'''

for d, fn in JOBS:
    src = (ROOT / "kaggle/v1_original" / d / fn).read_text()
    assert src.count("PATIENCE = 8\n") == 1
    src = src.replace("PATIENCE = 8\n", CONST, 1)
    sig = 'def fit_fold(model, Xtr, ytr, Xva, yva, stream_list, loss="coral", softtr=None):\n'
    assert src.count(sig) == 1, fn
    i = src.index(sig)
    body_start = src.index('"""', src.index('"""', i) + 3) + 4  # after docstring line
    src = (src[:i] + HELPER
           + sig.replace("softtr=None):", "softtr=None,\n             subj_tr=None):")
           + src[i + len(sig):body_start] + GUARD + src[body_start:])
    # every call site: pass the training-fold subject ids, taken from the index
    # variable used by the preceding `Xtr = to_torch(...)` line.
    out, last_var, n = [], None, 0
    for line in src.splitlines(keepends=True):
        mv = re.search(r'Xtr = to_torch\(\{s: corpus\[f"feat_\{s\}"\]\[(\w+)\]', line)
        if mv:
            last_var = mv.group(1)
        if re.search(r"= fit_fold\(model, Xtr, ytr, Xva, yva,", line):
            assert last_var, fn
            line = line.replace("fit_fold(model, Xtr, ytr, Xva, yva,",
                                f'fit_fold(model, Xtr, ytr, Xva, yva, subj_tr=corpus["subjects"][{last_var}], stream_list=', 1)
            line = line.replace("stream_list= stream_list,", "stream_list=stream_list,").replace("stream_list= all_present,", "stream_list=all_present,")
            n += 1
        out.append(line)
    src = "".join(out)
    assert n == 2, (fn, n)
    hdr = ("# === v2 NESTED: derived from v1 by tools/make_nested.py. Early stopping now uses an\n"
           "# === inner subject split of the training fold (GTAO_NESTED=0 restores v1 behaviour).\n")
    (ROOT / "kaggle/v2_nested" / d / fn).write_text(hdr + src)
    print("patched", d, fn, "call sites:", n)
