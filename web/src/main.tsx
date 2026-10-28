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

createRoot(document.getElementById('raiz')!).render(<App />);
