# Build paper/build/main_short.pdf (10-page version) on Windows with MiKTeX.
# Usage, from the bundle root:  powershell -ExecutionPolicy Bypass -File tools\build_short.ps1
$ErrorActionPreference = "Stop"
$bin = Join-Path $env:LOCALAPPDATA "Programs\MiKTeX\miktex\bin\x64"
$paper = Join-Path $PSScriptRoot "..\paper"
Set-Location $paper
New-Item -ItemType Directory -Force build | Out-Null

& "$bin\pdflatex.exe" -interaction=nonstopmode -output-directory=build main_short.tex | Out-Null
Copy-Item references.bib build\ -Force
Push-Location build
& "$bin\bibtex.exe" main_short
Pop-Location
& "$bin\pdflatex.exe" -interaction=nonstopmode -output-directory=build main_short.tex | Out-Null
& "$bin\pdflatex.exe" -interaction=nonstopmode -output-directory=build main_short.tex | Out-Null

Select-String -Path build\main_short.log -Pattern "undefined|Overfull|LaTeX Error"
Select-String -Path build\main_short.log -Pattern "Output written"
Write-Host "PDF: $(Resolve-Path build\main_short.pdf)"
