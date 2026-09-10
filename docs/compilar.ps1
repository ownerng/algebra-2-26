# Compila el informe sin make (Windows). Equivalente al Makefile.
Set-Location $PSScriptRoot
pdflatex informe.tex
biber informe
pdflatex informe.tex
pdflatex informe.tex
