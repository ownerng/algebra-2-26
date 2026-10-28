# Escáner LIDAR 2D: tareas frecuentes

.PHONY: web datos test informe limpiar

web:            ## Abre la interfaz en http://localhost:8000
	./iniciar-web.sh

datos:          ## Regenera las cifras, tablas y CSV del informe desde el simulador
	node experimentos/generar.mjs

test:           ## Pruebas de la matemática
	node --test tests/matematica.test.mjs

informe: datos  ## Compila docs/informe.pdf (requiere TeX Live con apa7 y biber)
	cd docs && pdflatex informe.tex && biber informe && pdflatex informe.tex && pdflatex informe.tex

limpiar:
	cd docs && rm -f *.aux *.log *.bbl *.bcf *.blg *.out *.run.xml
