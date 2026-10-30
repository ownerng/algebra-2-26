// Panel izquierdo: conectar DroidCam por WiFi (IP) o por cable USB (cámara del
// computador que crea el cliente de DroidCam). Al llegar la primera imagen de
// la cámara el panel se cierra solo.

import { useEffect, useState } from 'react';
import { camarasUsb, conectarCamara, type CamaraUsb } from '../lib/api.ts';

const CLAVE_IP = 'escaner-3d-droidcam-ip';
const CLAVE_VIA = 'escaner-3d-via';
const leer = (clave: string) => { try { return localStorage.getItem(clave) ?? ''; } catch { return ''; } };
const guardar = (clave: string, valor: string) => { try { localStorage.setItem(clave, valor); } catch { /* sin almacenamiento */ } };

type Via = 'wifi' | 'usb';

export function PanelCamara({ conectada, alConectar, cerrar }: { conectada: boolean; alConectar: () => void; cerrar: () => void }) {
  const [via, setVia] = useState<Via>(leer(CLAVE_VIA) === 'usb' ? 'usb' : 'wifi');
  const [ip, setIp] = useState(() => leer(CLAVE_IP));
  const [usb, setUsb] = useState<CamaraUsb[] | null>(null);
  const [ruta, setRuta] = useState('');
  const [estado, setEstado] = useState<'' | 'conectando' | 'error'>('');
  const [mensaje, setMensaje] = useState('');

  const buscarUsb = async () => {
    setUsb(null);
    try {
      const lista = await camarasUsb();
      setUsb(lista);
      // Se prefiere la cámara de DroidCam si está
      setRuta((r) => (lista.some((c) => c.ruta === r) ? r : (lista.find((c) => c.droidcam) ?? lista[0])?.ruta ?? ''));
    } catch (e) {
      setUsb([]); setEstado('error'); setMensaje((e as Error).message);
    }
  };

  useEffect(() => { if (via === 'usb') void buscarUsb(); }, [via]);

  const cambiarVia = (v: Via) => { setVia(v); guardar(CLAVE_VIA, v); setEstado(''); setMensaje(''); };

  const conectar = async (ev: React.FormEvent) => {
    ev.preventDefault();
    const destino = via === 'wifi' ? ip.trim() : ruta;
    if (via === 'wifi') guardar(CLAVE_IP, destino);
    setEstado('conectando'); setMensaje('');
    try {
      const e = await conectarCamara(destino);
      if (e.conectada) { setEstado(''); alConectar(); return; }
      setEstado('error'); setMensaje(e.error || 'no llegó imagen de la cámara');
    } catch (err) {
      setEstado('error'); setMensaje((err as Error).message);
    }
  };

  const ocupado = estado === 'conectando';
  const elegida = usb?.find((c) => c.ruta === ruta);

  return (
    <aside className="panel" aria-label="Cámara">
      <div className="panel-cabeza">
        <h2>Cámara</h2>
        <button type="button" className="cerrar" onClick={cerrar} aria-label="Cerrar">✕</button>
      </div>

      <div className="selector ancho" role="group" aria-label="Conexión">
        <button type="button" aria-pressed={via === 'wifi'} disabled={ocupado} onClick={() => cambiarVia('wifi')}>WiFi</button>
        <button type="button" aria-pressed={via === 'usb'} disabled={ocupado} onClick={() => cambiarVia('usb')}>USB</button>
      </div>

      <form onSubmit={conectar}>
        {via === 'wifi' ? (
          <>
            <label htmlFor="ip">IP de DroidCam</label>
            <input id="ip" inputMode="decimal" autoComplete="off" placeholder="192.168.1.5" value={ip}
              onChange={(ev) => setIp(ev.target.value)} disabled={ocupado} />
            <p className="ayuda">Abre DroidCam en el celular y copia la <b>WiFi IP</b>. El celular y el computador deben estar en la misma red.</p>
          </>
        ) : (
          <>
            <label htmlFor="usb">Cámara del computador</label>
            <div className="fila-campo">
              <select id="usb" value={ruta} onChange={(ev) => setRuta(ev.target.value)} disabled={ocupado || !usb?.length}>
                {usb === null && <option>Buscando…</option>}
                {usb?.length === 0 && <option>No hay cámaras</option>}
                {usb?.map((c) => <option key={c.ruta} value={c.ruta}>{c.nombre} ({c.ruta.replace('/dev/', '')})</option>)}
              </select>
              <button type="button" className="boton" onClick={() => void buscarUsb()} disabled={ocupado} aria-label="Buscar otra vez" title="Buscar otra vez">↻</button>
            </div>
            <p className="ayuda">
              Conecta el iPhone por cable y abre DroidCam en él, con la pantalla encendida. Aparecerá
              <b> iPhone por cable</b>; no hace falta WiFi ni instalar nada.
              {usb && !usb.some((c) => c.droidcam) && ' Todavía no aparece: revisa el cable, desbloquea el iPhone y pulsa ↻.'}
              {elegida && !elegida.droidcam && ' (Esta es una cámara del computador; también sirve para probar.)'}
            </p>
          </>
        )}
        <button type="submit" className="boton primario" disabled={ocupado || (via === 'wifi' ? !ip.trim() : !ruta)}>
          {ocupado ? 'Conectando…' : 'Conectar'}
        </button>
      </form>
      <p className={'estado ' + (conectada ? 'ok' : estado === 'error' ? 'mal' : '')} role="status">
        <i aria-hidden="true" />
        {conectada ? 'Cámara conectada' : estado === 'error' ? `No se conectó: ${mensaje}` : 'Sin cámara'}
      </p>
    </aside>
  );
}
