#!/bin/bash
# Abre la interfaz del escáner en http://localhost:8000
# (Web Serial y los módulos JS necesitan un servidor local, no funcionan con doble clic).
cd "$(dirname "$0")/web"
PUERTO=8000
( sleep 1; xdg-open "http://localhost:$PUERTO" >/dev/null 2>&1 ) &
echo "Interfaz en http://localhost:$PUERTO  (Ctrl+C para cerrar)"
python3 -m http.server "$PUERTO" --bind 127.0.0.1
