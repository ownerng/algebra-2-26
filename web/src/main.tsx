import { createRoot } from 'react-dom/client';
import '@fontsource/barlow-condensed/500.css';
import '@fontsource/barlow-condensed/600.css';
import '@fontsource/barlow-condensed/700.css';
import '@fontsource/ibm-plex-mono/400.css';
import '@fontsource/ibm-plex-mono/500.css';
import '@fontsource/newsreader/400.css';
import '@fontsource/newsreader/400-italic.css';
import 'katex/dist/katex.min.css';
import './estilos.css';
import { App } from './App.tsx';
import { escaneo } from './lib/escaneo.ts';
import { camara } from './lib/camara.ts';
import * as homografia from './lib/homografia.ts';

// Acceso desde la consola del navegador, para depurar y para las pruebas
(window as unknown as Record<string, unknown>).escaner3d = { escaneo, camara, homografia };

createRoot(document.getElementById('raiz')!).render(<App />);
