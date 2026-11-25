"""Phase 6 on Kaggle: rich Tchebichef re-extraction of DIPSER, downloaded from SciDB.

Traceable chain: official SciDB FTP -> this kernel -> per-zip feature files.
The feature code is NOT reimplemented: pc_pipeline/extract_features.py, _common.py and
kaggle/v2_nested/dipser_grid/gtao_grid.py are embedded verbatim (EMBEDDED below, written
by make_kernels.py; their sha256 is printed) and their process() is called per zip.

MODE "test": one zip -> download speed, frames per window, and the self-test against
the paper features in corpus_full.npz (leading Tchebichef coefficients must be equal).
MODE "shard": zips SHARD, SHARD + N_SHARDS, ... (sorted FTP list); per-zip outputs in
/kaggle/working/rich/. Stops starting downloads after STOP_AFTER_H.
FTP credentials come from Kaggle Secrets SCIDB_USER / SCIDB_PASS (never in code).
"""
import ftplib
import hashlib
import json
import multiprocessing as mp
import os
import queue
import shutil
import sys
import threading
import time
from pathlib import Path

import numpy as np

MODE = "test"            # set by make_kernels.py
SHARD, N_SHARDS = 0, 1   # set by make_kernels.py
SEQ_K, FACE_K = 12, 24   # set by make_kernels.py
WORKERS = 3
STOP_AFTER_H = 11.0
MAX_WAITING = 4          # downloaded zips waiting for extraction (disk guard)
HOST, PORT, ROOT = "ftp.scidb.cn", 2121, "/DIPSEER"
EMBEDDED = {}            # filled by make_kernels.py: relative path -> source text

OUT = Path("/kaggle/working") if os.path.isdir("/kaggle/working") else Path("v5_out")
RICH = OUT / "rich"
TMP = Path("/tmp/v5")
ZIPS, WORK, CODE = TMP / "zips", TMP / "work", TMP / "code"
START = time.time()


def log(msg):
    print(f"[{(time.time() - START) / 60:6.1f} min] {msg}", flush=True)


EMBEDDED_USER, EMBEDDED_PASS = None, None   # set by make_kernels.py from env (never in git)


def credentials():
    if EMBEDDED_USER and EMBEDDED_PASS:
        return EMBEDDED_USER, EMBEDDED_PASS
    try:
        from kaggle_secrets import UserSecretsClient
        client = UserSecretsClient()
        return client.get_secret("SCIDB_USER"), client.get_secret("SCIDB_PASS")
    except Exception as error:                      # local runs: environment variables
        user, password = os.environ.get("SCIDB_USER"), os.environ.get("SCIDB_PASS")
        if not user or not password:
            raise SystemExit(f"no SciDB credentials (Kaggle Secrets SCIDB_USER/SCIDB_PASS): {error}")
        return user, password


def connect(user, password):
    ftp = ftplib.FTP()
    ftp.connect(HOST, PORT, timeout=120)
    ftp.login(user, password)
    ftp.set_pasv(True)
    ftp.voidcmd("TYPE I")
    return ftp


def _names(ftp):
    """Entries of the current directory. This server accepts LIST only (not NLST/MLSD)."""
    lines = []
    ftp.retrlines("LIST", lines.append)
    return [line.split(None, 8)[-1] for line in lines if len(line.split(None, 8)) == 9]


def list_zips(user, password):
    """Walk ROOT/group_*/experiment_*/subject_*.zip by changing directory.

    Some servers reject NLST <path>; listing the current directory always works.
    """
    ftp = connect(user, password)
    log(f"FTP login ok; pwd {ftp.pwd()}; top level: {sorted(_names(ftp))[:20]}")
    root = None
    for candidate in (ROOT, ROOT.lstrip("/"), "/" + ROOT.lstrip("/").lower(), "."):
        try:
            ftp.cwd("/")
            ftp.cwd(candidate)
            if any(n.startswith("group_") for n in _names(ftp)):
                root = ftp.pwd()
                break
        except ftplib.error_perm:
            continue
    if root is None:
        raise SystemExit("could not find the group_* folders on the FTP server (see top level above)")
    log(f"dataset root: {root}")
    zips = []
    for group in sorted(n for n in _names(ftp) if n.startswith("group_")):
        ftp.cwd(f"{root}/{group}")
        for experiment in sorted(n for n in _names(ftp) if n.startswith("experiment_")):
            folder = f"{root}/{group}/{experiment}"
            ftp.cwd(folder)
            for base in sorted(_names(ftp)):
                if base.startswith("subject_") and base.endswith(".zip"):
                    zips.append((f"{folder}/{base}", ftp.size(base)))
    ftp.quit()
    return zips


def download(path, size, user, password, dest):
    """Resumable binary download with retries; returns MB/s."""
    for attempt in range(1, 9):
        try:
            have = dest.stat().st_size if dest.exists() else 0
            if have == size:
                return 0.0
            ftp = connect(user, password)
            ftp.cwd(str(Path(path).parent))          # RETR <basename>: no path argument
            t0 = time.time()
            with open(dest, "ab") as handle:
                ftp.retrbinary(f"RETR {Path(path).name}", handle.write, blocksize=1 << 20,
                               rest=have if have else None)
            ftp.quit()
            if dest.stat().st_size == size:
                return (size - have) / 1e6 / max(time.time() - t0, 1e-6)
        except Exception as error:
            log(f"  download retry {attempt} for {path}: {type(error).__name__}: {error}")
            time.sleep(min(60, 5 * attempt))
    raise RuntimeError(f"download failed: {path}")


def install_code():
    """Write the embedded PC pipeline + grid script and return the extract module."""
    for rel, text in EMBEDDED.items():
        target = CODE / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")
        log(f"code {rel}: sha256 {hashlib.sha256(text.encode('utf-8')).hexdigest()[:16]}")
    sys.path.insert(0, str(CODE / "pc_pipeline"))
    import extract_features as E
    return E


def stem_of(path):
    parts = Path(path).with_suffix("").parts[-3:]
    return "__".join(parts), "/".join(parts)


def self_test(rich_npz, corpus_path):
    """Leading coefficients of the rich features == paper features (same keys)."""
    r = np.load(rich_npz)
    base = np.load(corpus_path)
    position = {k: i for i, k in enumerate(base["keys"].astype(str))}
    idx = np.array([position.get(k, -1) for k in r["keys"].astype(str)])
    found = idx >= 0
    ok = found & ((r["n_frames"] >= SEQ_K) | (SEQ_K == 6))
    report = {"windows": int(len(idx)), "matched_keys": int(found.sum()), "compared": 0,
              "max_rel_diff": 0.0}
    # physio is excluded: channels with < SEQ_K samples per window are edge-padded to
    # SEQ_K, so a higher order changes even the leading coefficients (by construction)
    for stream, n_values, face in (("face", FACE_K ** 2, FACE_K), ("facemesh", 478 * 3, None),
                                   ("head_pose", 3, None)):
        rows = ok & r[f"avail_{stream}"]
        rows[found] &= base[f"avail_{stream}"][idx[found]]
        if not rows.any():
            continue
        x = r[f"feat_{stream}"][rows].reshape(int(rows.sum()), n_values, SEQ_K)[:, :, :6]
        if face:
            x = x.reshape(len(x), face, face, 6)[:, :16, :16, :]
        b = base[f"feat_{stream}"][idx[rows]]
        diff = np.abs(x.reshape(len(x), -1) - b).max() / (np.abs(b).max() + 1e-9)
        report["compared"] += int(rows.sum())
        report["max_rel_diff"] = max(report["max_rel_diff"], float(diff))
    report["passed"] = report["compared"] > 0 and report["max_rel_diff"] <= 1e-4
    return report


def find_corpus():
    for folder, _, names in os.walk("/kaggle/input"):
        if "corpus_full.npz" in names:
            return Path(folder) / "corpus_full.npz"
    return None


def main():
    for d in (RICH, ZIPS, WORK):
        d.mkdir(parents=True, exist_ok=True)
    E = install_code()
    grid_path = str(CODE / "kaggle/v2_nested/dipser_grid/gtao_grid.py")
    user, password = credentials()
    zips = list_zips(user, password)
    log(f"FTP list: {len(zips)} subject zips, {sum(s for _, s in zips) / 1e9:.1f} GB")
    if MODE == "test":
        todo = zips[:1]
    else:
        todo = zips[SHARD::N_SHARDS]
    todo = [(p, s) for p, s in todo if not (RICH / f"{stem_of(p)[0]}.npz").exists()]
    log(f"mode {MODE}, shard {SHARD}/{N_SHARDS}: {len(todo)} zips, "
        f"{sum(s for _, s in todo) / 1e9:.1f} GB, seq-k {SEQ_K}, face-k {FACE_K}")

    ready = queue.Queue(maxsize=MAX_WAITING)
    speeds = []

    def downloader():
        for path, size in todo:
            if (time.time() - START) / 3600 > STOP_AFTER_H:
                log("time guard: no new downloads")
                break
            stem, key = stem_of(path)
            dest = ZIPS / f"{stem}.zip"
            try:
                speed = download(path, size, user, password, dest)
                speeds.append(speed)
                log(f"downloaded {stem} ({size / 1e9:.2f} GB, {speed:.1f} MB/s)")
                ready.put((str(dest), key, stem))
            except Exception as error:
                log(f"SKIP {stem}: {error}")
        ready.put(None)

    thread = threading.Thread(target=downloader, daemon=True)
    thread.start()
    lock = mp.Manager().Lock()
    results = []
    with mp.Pool(WORKERS, initializer=E.init_worker,
                 initargs=(lock, grid_path, SEQ_K, FACE_K)) as pool:
        pending = []
        while True:
            item = ready.get()
            if item is None:
                break
            dest, key, stem = item
            while sum(not p.ready() for p in pending) >= WORKERS + 1:   # disk guard
                time.sleep(2)
            job = (dest, key, stem, str(RICH), str(WORK))
            pending.append(pool.apply_async(E.process, (job,),
                                            callback=lambda r, d=dest: (Path(d).unlink(missing_ok=True),
                                                                        results.append(r))))
        for p in pending:
            p.wait()
    ok = [r for r in results if r[1] == "ok"]
    bad = [r for r in results if r[1] != "ok"]
    for stem, _, info in bad:
        log(f"FAILED {stem}: {str(info).splitlines()[0]}")
    summary = {"mode": MODE, "shard": SHARD, "n_shards": N_SHARDS, "seq_k": SEQ_K,
               "face_k": FACE_K, "zips_ok": len(ok), "zips_failed": [r[0] for r in bad],
               "zips_planned": len(todo), "mean_MBps": float(np.mean(speeds)) if speeds else None,
               "wall_h": round((time.time() - START) / 3600, 2)}
    if ok:
        frames = np.concatenate([np.load(RICH / f"{s}.npz")["n_frames"] for s, _, _ in ok])
        summary["frames_per_window"] = [int(frames.min()), float(np.median(frames)), int(frames.max())]
        summary["share_ge_seq_k"] = float((frames >= SEQ_K).mean())
        corpus = find_corpus()
        if corpus is not None:
            summary["self_test"] = [self_test(RICH / f"{s}.npz", corpus) | {"zip": s}
                                    for s, _, _ in ok[:3]]
    (OUT / f"summary_{MODE}_{SHARD}.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))
    shutil.rmtree(TMP, ignore_errors=True)


if __name__ == "__main__":
    main()
