# Unattended Phase 3 run: keep the download alive, extract features as zips arrive,
# count the labels, and build the corpus files once everything is downloaded.
#
#   schtasks /create /tn gtao-overnight /sc onlogon /rl highest /f /tr "powershell -NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File D:\Study\phd\gtao-workbench\pc_pipeline\overnight.ps1"
#   schtasks /run /tn gtao-overnight
#
# Task Scheduler, not Start-Process from a Claude Code shell: processes started from the
# shell can die with the session. The logon trigger restarts it after a reboot; the task
# deletes itself when everything is done.
# Safe to start while a pull.ps1 is already running: it waits for that rclone to exit.
# Safe to restart at any time: pull and extraction are both resumable.
# Progress: C:\dipser_features\overnight.log (steps), overnight_download.log (download),
# extract.log (extraction/merge output). When everything is done it writes
# C:\dipser_features\OVERNIGHT_DONE.

param(
    [string]$Data = "D:\DIPSER",
    [string]$Features = "C:\dipser_features",
    [string]$Work = "C:\dipser_work",
    [int]$Workers = 5,
    [int]$MaxDownloadRounds = 20
)

$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$python = "C:\gtao_env\Scripts\python.exe"
New-Item -ItemType Directory -Force -Path $Features, $Work | Out-Null
$log = Join-Path $Features "overnight.log"
$downloadLog = Join-Path $Features "overnight_download.log"
$pyLog = Join-Path $Features "extract.log"
$downloadDone = Join-Path $Features "DOWNLOAD_DONE"
Remove-Item $downloadDone, (Join-Path $Features "OVERNIGHT_DONE") -ErrorAction SilentlyContinue

function Log($message) {
    $line = "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') $message"
    Add-Content -Path $log -Value $line
    Write-Host $line
}

# Download loop, in its own process so extraction runs alongside it.
$downloadScript = {
    param($here, $Data, $downloadDone, $log, $MaxDownloadRounds)
    function Log($m) { Add-Content -Path $log -Value "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') [download] $m" }
    while (Get-Process rclone -ErrorAction SilentlyContinue) {
        Log "an rclone is already running; waiting for it to exit"
        Start-Sleep -Seconds 300
    }
    for ($round = 1; $round -le $MaxDownloadRounds; $round++) {
        # rclone restarts interrupted files under a new random name: drop the orphans
        Get-ChildItem $Data -Recurse -Filter *.partial -ErrorAction SilentlyContinue | Remove-Item -Force
        foreach ($group in "group_02", "group_03", "group_01") {
            Log "round $round : pulling $group"
            & (Join-Path $here "pull.ps1") -Dest $Data -Include "$group/experiment_*" | Out-Null
        }
        rclone check "scidb:/DIPSEER" $Data --include "group_*/experiment_*/subject_*.zip" `
            --size-only --one-way 2>&1 | Out-Null
        if ($LASTEXITCODE -eq 0) {
            Log "all zips present and sizes match the server"
            New-Item -ItemType File -Force -Path $downloadDone | Out-Null
            return
        }
        Log "round $round : some zips missing or wrong size; retrying in 10 min"
        Start-Sleep -Seconds 600
    }
    Log "GAVE UP after $MaxDownloadRounds rounds (credentials expired?)"
    New-Item -ItemType File -Force -Path $downloadDone | Out-Null
}
$job = Start-Job -ScriptBlock $downloadScript -ArgumentList $here, $Data, $downloadDone, $downloadLog, $MaxDownloadRounds
Log "started; download job $($job.Id)"

Set-Location $here
while ($true) {
    # Stall watchdog: dropped FTP connections can leave rclone hung for hours without
    # timing out (seen 2026-11-18 05:44). Killing it lets the download loop retry.
    $rclone = Get-Process rclone -ErrorAction SilentlyContinue
    if ($rclone) {
        $latest = (Get-ChildItem $Data -Recurse -File -ErrorAction SilentlyContinue |
                   Sort-Object LastWriteTime -Descending | Select-Object -First 1).LastWriteTime
        if ($latest -and $latest -lt (Get-Date).AddMinutes(-30)) {
            Log "download stalled (nothing written since $latest); killing rclone"
            $rclone | Stop-Process -Force
        }
    }
    $finalPass = Test-Path $downloadDone      # read BEFORE extracting: no zip can arrive later
    Log "extraction pass (final=$finalPass)"
    & $python extract_features.py --data $Data --out $Features --work $Work --workers $Workers *>> $pyLog
    & $python count_labels.py --data $Data --out (Join-Path $Features "label_counts.csv") *> (Join-Path $Features "label_counts.txt")
    if ($finalPass) { break }
    Start-Sleep -Seconds 900
}

# One retry for zips that failed (their .npz is missing, so extraction picks them up again).
Log "retry pass for failures"
& $python extract_features.py --data $Data --out $Features --work $Work --workers $Workers *>> $pyLog

Log "merging"
& $python merge_features.py --features $Features --out (Join-Path $Features "corpus_full.npz") *>> $pyLog
& $python merge_features.py --features $Features --out (Join-Path $Features "corpus_paper_slice.npz") `
    --only group_01/experiment_01 group_01/experiment_05 *>> $pyLog

Receive-Job $job *>> $downloadLog
New-Item -ItemType File -Force -Path (Join-Path $Features "OVERNIGHT_DONE") | Out-Null
Log "ALL DONE"
schtasks /delete /tn gtao-overnight /f | Out-Null
