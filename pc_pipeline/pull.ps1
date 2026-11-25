# Download the DIPSER per-student zips from SciDB FTP to the HDD (resumable).
#
#   powershell -ExecutionPolicy Bypass -File pull.ps1                       # everything
#   powershell -ExecutionPolicy Bypass -File pull.ps1 -Include "group_01/experiment_*"   # one group
#
# Needs an rclone remote called "scidb" (see README_PC.md, step 3).
# Re-running the same command resumes: finished files are skipped, and rclone writes
# into *.partial files, so extract_features.py never sees a half-downloaded zip.
# The context_cam_NN.zip files (~250 GB, not used by gTAO) are NOT downloaded.

param(
    [string]$Dest = "D:\DIPSER",
    [string]$Include = "group_*/experiment_*",   # e.g. "group_01/experiment_*" or "group_02/experiment_03"
    [int]$Transfers = 6
)

$pattern = "$Include/subject_*.zip"

New-Item -ItemType Directory -Force -Path $Dest | Out-Null
$log = Join-Path $Dest "pull.log"
Write-Host "Downloading scidb:/DIPSEER/$pattern -> $Dest  (log: $log)"

rclone copy "scidb:/DIPSEER" $Dest `
    --include $pattern `
    --transfers $Transfers --checkers 8 --ftp-concurrency ($Transfers + 2) `
    --retries 20 --low-level-retries 50 --timeout 10m --contimeout 2m `
    --stats 30s -P --log-file $log --log-level INFO

if ($LASTEXITCODE -ne 0) {
    Write-Host "rclone stopped with errors. Just run the same command again to resume." -ForegroundColor Yellow
} else {
    Write-Host "Done. Verifying sizes against the server..."
    rclone check "scidb:/DIPSEER" $Dest --include $pattern --size-only --one-way
}
