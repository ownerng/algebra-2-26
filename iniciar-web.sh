#!/bin/bash
# Abre el escáner 3D: el servidor en Python (http://localhost:8000: cámara,
# segmentación y malla) y la página (http://localhost:3000).
cd "$(dirname "$0")"
[ -d servidor/.venv ] || python3 -m venv servidor/.venv
# Instala lo que falte (rápido si ya está todo)
servidor/.venv/bin/pip install -q -r servidor/requirements.txt
[ -d web/node_modules ] || (cd web && bun install)
servidor/.venv/bin/uvicorn app:app --app-dir servidor --host 0.0.0.0 --port 8000 &
SERVIDOR=$!
trap 'kill $SERVIDOR 2>/dev/null' EXIT
( sleep 2; xdg-open "http://localhost:3000" >/dev/null 2>&1 ) &
echo "Interfaz en http://localhost:3000  (Ctrl+C para cerrar)"
cd web && bun run dev
