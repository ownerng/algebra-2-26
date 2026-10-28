# Escáner LIDAR 2D: tareas frecuentes

.PHONY: web datos test graficas informe limpiar

web:            ## Abre la interfaz en http://localhost:3000 (Bun)
	./iniciar-web.sh

datos:          ## Regenera las cifras, tablas y CSV del informe desde el simulador
	bun experimentos/generar.mjs

test:           ## Pruebas de la matemática
	bun test tests/

graficas: datos ## Compila las figuras de docs/graficas/*.tex a PDF
	cd docs && for f in graficas/*.tex; do pdflatex -output-directory graficas $$f; done

informe: graficas  ## Compila docs/informe.pdf (requiere TeX Live con apa7 y apacite)
	cd docs && pdflatex informe.tex && bibtex informe && pdflatex informe.tex && pdflatex informe.tex

limpiar:
	cd docs && rm -f *.aux *.log *.bbl *.bcf *.blg *.out *.run.xml graficas/*.aux graficas/*.log
