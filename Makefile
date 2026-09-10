informe:
	cd docs && pdflatex informe.tex
	cd docs && biber informe
	cd docs && pdflatex informe.tex
	cd docs && pdflatex informe.tex

clean:
	cd docs && rm -f *.aux *.log *.bbl *.bcf *.blg *.out *.run.xml
