#!/bin/bash
# Abre la interfaz del escáner en http://localhost:3000 y el puente USB
# (ws://localhost:3001), que deja usar el ESP32 desde cualquier navegador.
cd "$(dirname "$0")/web"
[ -d node_modules ] || bun install
bun puente.ts &
PUENTE=$!
trap 'kill $PUENTE 2>/dev/null' EXIT
( sleep 1.5; xdg-open "http://localhost:3000" >/dev/null 2>&1 ) &
echo "Interfaz en http://localhost:3000  (Ctrl+C para cerrar)"
bun run dev
