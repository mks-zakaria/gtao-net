# DIPSER download and feature extraction on the Windows PC

PC: i5-9500, GTX 1650, 16 GB RAM. Zips go on the **HDD (D:)**; features and the
temporary unzip folder go on the **SSD (C:)**. Adjust the drive letters if yours differ.

What gets downloaded: only the per-student `subject_NN.zip` files, ~500-540 GB
(est.). The `context_cam_NN.zip` files (~250 GB) are skipped.
Output: one small `.npz` of features per student, then one corpus file (~1-2 GB) for the M5.

## 0. Get the repo on the PC

```powershell
winget install Git.Git GitHub.cli
gh auth login                      # account mks-zakaria, HTTPS, browser login
gh repo clone mks-zakaria/gtao-workbench C:\gtao-workbench
cd C:\gtao-workbench
git config user.name mks-zakaria
git config user.email mks-zakaria@users.noreply.github.com
```

Then start Claude Code in `C:\gtao-workbench` and say "continue": it reads `CLAUDE.md`
and `STATUS.md`.

## 1. Windows preparation (once)

Open PowerShell **as administrator**:

```powershell
winget install Rclone.Rclone
winget install Python.Python.3.12
# never sleep while plugged in (the download takes hours to days)
powercfg /change standby-timeout-ac 0
# long file paths inside the zips
New-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem" -Name LongPathsEnabled -Value 1 -PropertyType DWORD -Force
# stop Defender from scanning 500 GB of zips and every extracted frame
Add-MpPreference -ExclusionPath "D:\DIPSER", "C:\dipser_work", "C:\dipser_features"
```

Close and reopen PowerShell (normal user is fine from here).

## 2. Python environment

```powershell
py -3.12 -m venv C:\gtao_env
C:\gtao_env\Scripts\pip install torch mpmath scikit-learn pillow numpy
cd C:\gtao-workbench\pc_pipeline
C:\gtao_env\Scripts\python self_test.py
```

`self_test.py` must end with **SELF-TEST PASSED**. It proves the PC produces exactly the
same features as the paper's code. Do not continue if it fails.

## 3. FTP access (credentials expire after 7 days)

1. Request fresh credentials on the SciDB DIPSER page
   (https://www.scidb.cn/en/detail?dataSetId=7856c716c0cc4589a23ee4a23d8a0893).
2. As soon as they arrive, create the remote (type the real user and password):

```powershell
rclone config create scidb ftp host=ftp.scidb.cn port=2121 user=YOUR_USER pass=YOUR_PASS --obscure
rclone lsd scidb:/DIPSEER          # must list group_01, group_02, group_03
```

## 4. Measure before the big pull

```powershell
rclone size scidb:/DIPSEER --include "group_*/experiment_*/subject_*.zip"
rclone copy scidb:/DIPSEER/group_01/experiment_02/subject_01.zip D:\DIPSER\group_01\experiment_02 -P
```

Write down the total size and the MB/s shown. Expected time for ~540 GB:
25 MB/s (your fibre's maximum) ~ 6 h; 5 MB/s ~ 30 h; 1 MB/s ~ 6 days (too close to
the 7-day credential expiry: in that case tell me and we pick a subset).

## 5. Download (resumable)

```powershell
cd C:\gtao-workbench\pc_pipeline
powershell -ExecutionPolicy Bypass -File pull.ps1 -Include "group_01/experiment_*"
powershell -ExecutionPolicy Bypass -File pull.ps1 -Include "group_02/experiment_*"
powershell -ExecutionPolicy Bypass -File pull.ps1 -Include "group_03/experiment_*"
```

If it stops (network, reboot), run the same line again: finished files are skipped.
Launched from a Claude Code shell, rclone dies when that session closes (this happened
on 2026-11-17). Launch it detached instead:

```powershell
Start-Process powershell -WindowStyle Minimized -ArgumentList "-NoProfile","-ExecutionPolicy","Bypass","-Command","& 'D:\Study\phd\gtao-workbench\pc_pipeline\pull.ps1' -Include 'group_02/experiment_*'"
```

Delete leftover `*.partial` files after an interrupted run (rclone restarts them under a
new name).
More parallel downloads: add `-Transfers 10` (only if the per-file speed is low).

**Unattended:** `overnight.ps1` does steps 5-8 by itself (download with retries,
extraction as zips arrive, label counts, merges). See its header for how to start it.

## 6. Count the labels (any time, even during the download)

```powershell
C:\gtao_env\Scripts\python count_labels.py --data D:\DIPSER --out C:\dipser_features\label_counts.csv
```

**Send me the "Per experiment" and "TOTAL" lines.** The number of HIGH windows decides
whether the 3-class task is viable or whether we pre-declare the binary analysis.

## 7. Extract features (can run while later groups are still downloading)

```powershell
C:\gtao_env\Scripts\python extract_features.py --data D:\DIPSER --out C:\dipser_features --work C:\dipser_work --workers 5
```

- Resumable: re-run it after more zips arrive; done zips are skipped.
- Only one worker reads the HDD at a time; the other cores encode in parallel.
- It prints an ETA. Failures go to `C:\dipser_features\failed.log` (re-run to retry).
- Peak SSD use: ~5 unzipped students at a time (a few GB) plus the features.

## 8. Build the corpus and move it to the M5

```powershell
C:\gtao_env\Scripts\python merge_features.py --features C:\dipser_features --out C:\dipser_features\corpus_full.npz
# the paper's original 40 zips only (checks we reproduce the paper's corpus)
C:\gtao_env\Scripts\python merge_features.py --features C:\dipser_features --out C:\dipser_features\corpus_paper_slice.npz --only group_01/experiment_01 group_01/experiment_05
```

Copy both `.npz` files and their `.json` reports to the M5 (USB stick or network).
**Keep the zips on D:**: if Phase 1 changes the face settings, we re-extract without
downloading again.

## What NOT to do

- Do not delete the zips after extraction.
- Do not change `gtao_grid.py` on the PC: the features must come from the same code as the
  paper (each `.json` report records its SHA-256).
- Do not train on the PC: the models are too small to benefit from the GTX 1650. The GPU's
  job comes later (learned face-encoder baseline).
