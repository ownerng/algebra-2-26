#!/bin/bash
# Abre la interfaz del escáner en http://localhost:3000 con Bun.
# Web Serial necesita un servidor local (localhost); no funciona con doble clic.
cd "$(dirname "$0")/web"
[ -d node_modules ] || bun install
( sleep 1.5; xdg-open "http://localhost:3000" >/dev/null 2>&1 ) &
echo "Interfaz en http://localhost:3000  (Ctrl+C para cerrar)"
bun run dev
