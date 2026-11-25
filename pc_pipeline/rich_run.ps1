# Unattended Phase 6 run: rich Tchebichef re-extraction of the full DIPSER corpus.
#
# MORNING, ONE COMMAND (from the repo folder, after git pull):
#   powershell -NoProfile -ExecutionPolicy Bypass -File pc_pipeline\rich_run.ps1 -Start
# -Start registers the hidden scheduled task "gtao-rich" (also restarts it after a reboot
# or logon), starts it, and returns immediately. You can close the window and leave.
# A normal (non-admin) PowerShell window is enough.
# To see progress later:  Get-Content C:\dipser_features_rich\rich.log -Tail 15
# To stop it:             schtasks /end /tn gtao-rich ; schtasks /delete /tn gtao-rich /f
#
# What the task does:
#   1. probe: extract group_01/experiment_01 at temporal order 12, face order 24, then
#      check_rich.py (leading coefficients must equal the paper features).
#      Exit 2 (< 12 frames per window): order = median frames if >= 7, else 6; re-probe.
#      Exit 1: writes RICH_FAILED and stops.
#   2. full extraction of all 483 zips with the chosen orders (resumable) + 1 retry pass
#   3. merge -> corpus_rich.npz, check_rich.py on everything
#   4. writes RICH_DONE (or RICH_FAILED) and deletes the task.
# Progress: C:\dipser_features_rich\rich.log (steps), rich_python.log (python output).
# Keeps the PC awake while it runs. Safe to restart at any time (everything resumes).

param(
    [switch]$Start,
    [string]$Data = "D:\DIPSER",
    [string]$Base = "C:\dipser_features",
    [string]$Rich = "C:\dipser_features_rich",
    [string]$Work = "C:\dipser_work",
    [int]$Workers = 5,
    [int]$FaceK = 24,
    [int]$SeqK = 12
)

$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$task = "gtao-rich"

if ($Start) {
    if ($here -match " ") { Write-Host "ERROR: the repo path contains a space: $here"; exit 1 }
    $cmd = "powershell -NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File $here\rich_run.ps1"
    schtasks /create /tn $task /sc onlogon /f /tr $cmd
    if ($LASTEXITCODE -ne 0) { Write-Host "ERROR: could not create the scheduled task (see above)"; exit 1 }
    schtasks /run /tn $task
    if ($LASTEXITCODE -ne 0) { Write-Host "ERROR: could not start the scheduled task (see above)"; exit 1 }
    Write-Host "Started scheduled task '$task'. Progress: $Rich\rich.log"
    Write-Host "Finished when $Rich\RICH_DONE exists (or RICH_FAILED). You can leave now."
    exit 0
}

$python = "C:\gtao_env\Scripts\python.exe"
New-Item -ItemType Directory -Force -Path $Rich, $Work | Out-Null
$log = Join-Path $Rich "rich.log"
$pyLog = Join-Path $Rich "rich_python.log"
$done = Join-Path $Rich "RICH_DONE"
$failed = Join-Path $Rich "RICH_FAILED"
$orderFile = Join-Path $Rich "ORDERS.txt"
Remove-Item $done, $failed -ErrorAction SilentlyContinue

function Log($message) {
    $line = "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') $message"
    Add-Content -Path $log -Value $line
    Write-Host $line
}

function Finish($ok, $message) {
    Log $message
    if ($ok) { Set-Content -Path $done -Value $message } else { Set-Content -Path $failed -Value $message }
    schtasks /delete /tn $task /f | Out-Null
    exit 0
}

function Py([string[]]$arguments) {
    & $python @arguments 2>&1 | Out-File -Append -Encoding utf8 $pyLog
    return $LASTEXITCODE
}

function Check([int]$seq) {
    $out = & $python (Join-Path $here "check_rich.py") --base $Base --rich $Rich --seq-k $seq --face-k $FaceK 2>&1
    $code = $LASTEXITCODE
    $out | Out-File -Append -Encoding utf8 $pyLog
    $text = ($out | Out-String)
    foreach ($line in ($out | Select-String -Pattern "frames per window|windows compared|SELF-TEST")) { Log "  $line" }
    return @{ code = $code; text = $text }
}

# keep the machine awake while this process runs (ES_CONTINUOUS | ES_SYSTEM_REQUIRED)
Add-Type -Namespace Win32 -Name Power -MemberDefinition '[DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint esFlags);'
[Win32.Power]::SetThreadExecutionState([uint32]2147483649) | Out-Null   # 0x80000001

Log "=== rich run started (data $Data, base $Base, out $Rich, workers $Workers)"
if (-not (Test-Path $python)) { Finish $false "python not found at $python" }
if (-not (Test-Path (Join-Path $Base "corpus_full.npz"))) { Finish $false "base features missing in $Base" }

$extract = Join-Path $here "extract_features.py"

# ---- 1. probe (skipped when a previous run already fixed the orders) ----
if (Test-Path $orderFile) {
    $SeqK = [int](Get-Content $orderFile)
    Log "orders already chosen by an earlier probe: seq-k $SeqK, face-k $FaceK"
} else {
    $probeStart = Get-Date
    Log "probe: group_01/experiment_01 at seq-k $SeqK, face-k $FaceK"
    Py @($extract, "--data", $Data, "--out", $Rich, "--work", $Work, "--workers", $Workers,
         "--seq-k", $SeqK, "--face-k", $FaceK, "--only", "group_01/experiment_01") | Out-Null
    $n = (Get-ChildItem $Rich -Filter "group_*.npz").Count
    Log ("probe extracted {0} zips in {1:N1} min" -f $n, ((Get-Date) - $probeStart).TotalMinutes)
    $check = Check $SeqK
    $median = 0
    if ($check.text -match "median ([0-9.]+)") { $median = [double]$Matches[1] }
    # the order must be supported by the TYPICAL window, not just by some windows
    if ($check.code -eq 2 -or ($check.code -eq 0 -and $median -lt $SeqK)) {
        $new = if ($median -ge 7) { [int][math]::Floor($median) } else { 6 }
        Log "median frames per window $median < seq-k $SeqK: re-probing at seq-k $new"
        Get-ChildItem $Rich -Filter "group_*" | Remove-Item -Force
        $SeqK = $new
        Py @($extract, "--data", $Data, "--out", $Rich, "--work", $Work, "--workers", $Workers,
             "--seq-k", $SeqK, "--face-k", $FaceK, "--only", "group_01/experiment_01") | Out-Null
        $check = Check $SeqK
    }
    if ($check.code -ne 0) { Finish $false "probe self-test failed (exit $($check.code)); see $pyLog" }
    Set-Content -Path $orderFile -Value $SeqK
    Log "probe PASSED: using seq-k $SeqK, face-k $FaceK"
}

# ---- 2. full extraction (resumable) + one retry pass ----
for ($pass = 1; $pass -le 2; $pass++) {
    $t0 = Get-Date
    Log "extraction pass $pass (all zips; finished ones are skipped)"
    Py @($extract, "--data", $Data, "--out", $Rich, "--work", $Work, "--workers", $Workers,
         "--seq-k", $SeqK, "--face-k", $FaceK) | Out-Null
    $n = (Get-ChildItem $Rich -Filter "group_*.npz").Count
    Log ("pass {0}: {1} zips extracted ({2:N1} h)" -f $pass, $n, ((Get-Date) - $t0).TotalHours)
    if ($n -ge 483) { break }
}
$n = (Get-ChildItem $Rich -Filter "group_*.npz").Count
if ($n -lt 483) { Log "WARNING: only $n of 483 zips extracted; see $Rich\failed.log" }

# ---- 3. merge + full self-test ----
Log "merging -> corpus_rich.npz"
$code = Py @((Join-Path $here "merge_features.py"), "--features", $Rich, "--out", (Join-Path $Rich "corpus_rich.npz"))
if ($code -ne 0) { Finish $false "merge failed (exit $code); see $pyLog" }
$check = Check $SeqK
if ($check.code -ne 0) { Finish $false "full self-test failed (exit $($check.code)); see $pyLog" }
$size = "{0:N2} GB" -f ((Get-Item (Join-Path $Rich "corpus_rich.npz")).Length / 1GB)
Finish $true "RICH DONE: $n/483 zips, seq-k $SeqK, face-k $FaceK, corpus_rich.npz $size. Put corpus_rich.npz + corpus_rich.json on Google Drive."
