# Escáner 3D: tareas frecuentes

.PHONY: web datos test graficas informe limpiar

web:            ## Servidor Python (8000) e interfaz en http://localhost:3000
	./iniciar-web.sh

datos:          ## Regenera las cifras, tablas y CSV del informe (simulador y mediciones reales)
	bun experimentos/generar3d.ts
	bun experimentos/generar_reales3d.ts

test:           ## Pruebas de la matemática (TypeScript y Python)
	bun test tests/
	cd servidor && .venv/bin/pytest -q

graficas: datos ## Compila las figuras de docs/graficas/*.tex a PDF
	cd docs && for f in graficas/*.tex; do pdflatex -output-directory graficas $$f; done

informe: graficas  ## Compila docs/informe.pdf (requiere TeX Live con apa7 y apacite)
	cd docs && pdflatex informe.tex && bibtex informe && pdflatex informe.tex && pdflatex informe.tex

limpiar:
	cd docs && rm -f *.aux *.log *.bbl *.bcf *.blg *.out *.run.xml graficas/*.aux graficas/*.log
