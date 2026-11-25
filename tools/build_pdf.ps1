# Build paper/build/main.pdf on Windows with MiKTeX (no Perl/latexmk needed).
# Usage, from the bundle root:  powershell -ExecutionPolicy Bypass -File tools\build_pdf.ps1
# Tables are NOT regenerated here; run the python steps of `make tables` first if results changed.
$ErrorActionPreference = "Stop"
$bin = Join-Path $env:LOCALAPPDATA "Programs\MiKTeX\miktex\bin\x64"
$paper = Join-Path $PSScriptRoot "..\paper"
Set-Location $paper
New-Item -ItemType Directory -Force build | Out-Null

# pdflatex -> bibtex -> pdflatex -> pdflatex, the same sequence latexmk runs
& "$bin\pdflatex.exe" -interaction=nonstopmode -output-directory=build main.tex | Out-Null
Copy-Item references.bib build\ -Force
Push-Location build
& "$bin\bibtex.exe" main
Pop-Location
& "$bin\pdflatex.exe" -interaction=nonstopmode -output-directory=build main.tex | Out-Null
& "$bin\pdflatex.exe" -interaction=nonstopmode -output-directory=build main.tex | Out-Null

Select-String -Path build\main.log -Pattern "undefined|Overfull|LaTeX Error"
Write-Host "PDF: $(Resolve-Path build\main.pdf)"
