// Foto → modelo 3D con TRELLIS 2 (Microsoft, pesos abiertos con licencia MIT),
// ejecutado en fal.ai. El puente guarda la clave (FAL_KEY en web/.env, que no
// se sube a GitHub) para que nunca quede en la página.
//
//   POST /ia3d/clave   {clave}         guarda la clave
//   GET  /ia3d/listo                   ¿hay clave?
//   POST /ia3d/crear   {imagen}        envía la foto (data URI); devuelve {id}
//   GET  /ia3d/estado?id=…             {estado, posicion?, glb?}
//   GET  /ia3d/modelo?url=…            reenvía el .glb (evita el bloqueo CORS)

const MODELO = 'fal-ai/trellis-2';
const COLA = `https://queue.fal.run/${MODELO}`;
const ARCHIVO_ENV = new URL('./.env', import.meta.url).pathname;
const CORS = { 'Access-Control-Allow-Origin': '*', 'Access-Control-Allow-Headers': 'Content-Type', 'Cache-Control': 'no-store' };

const json = (d: unknown, status = 200) => new Response(JSON.stringify(d), { status, headers: { ...CORS, 'Content-Type': 'application/json' } });
const clave = () => process.env.FAL_KEY?.trim() ?? '';

async function guardarClave(nueva: string) {
  const archivo = Bun.file(ARCHIVO_ENV);
  const lineas = (await archivo.exists() ? await archivo.text() : '').split('\n').filter((l) => l.trim() && !l.startsWith('FAL_KEY='));
  lineas.push(`FAL_KEY=${nueva}`);
  await Bun.write(ARCHIVO_ENV, lineas.join('\n') + '\n');
  process.env.FAL_KEY = nueva;
}

async function fal(url: string, init?: RequestInit) {
  const r = await fetch(url, { ...init, headers: { Authorization: `Key ${clave()}`, 'Content-Type': 'application/json', ...init?.headers } });
  const texto = await r.text();
  let datos: Record<string, unknown> = {};
  try { datos = JSON.parse(texto); } catch { datos = { detail: texto.slice(0, 300) }; }
  if (!r.ok) {
    const detalle = typeof datos.detail === 'string' ? datos.detail : JSON.stringify(datos.detail ?? datos).slice(0, 300);
    throw new Error(r.status === 401 || r.status === 403 ? `la clave de fal.ai no es válida (${detalle})` : r.status === 402 || /balance|credit|locked/i.test(detalle) ? `sin saldo en fal.ai (${detalle})` : `fal.ai respondió ${r.status}: ${detalle}`);
  }
  return datos;
}

export async function ia3dHttp(req: Request, url: URL): Promise<Response> {
  if (req.method === 'OPTIONS') return new Response(null, { headers: { ...CORS, 'Access-Control-Allow-Methods': 'GET, POST' } });
  try {
    if (url.pathname === '/ia3d/listo') return json({ clave: !!clave(), modelo: MODELO });

    if (url.pathname === '/ia3d/clave' && req.method === 'POST') {
      const { clave: nueva } = (await req.json()) as { clave?: string };
      if (!nueva || !/^[\w:.-]{20,200}$/.test(nueva.trim())) return json({ error: 'clave con formato inválido' }, 400);
      await guardarClave(nueva.trim());
      return json({ ok: true });
    }

    if (!clave()) return json({ error: 'falta la clave de fal.ai (ponla en Ajustes)' }, 400);

    if (url.pathname === '/ia3d/crear' && req.method === 'POST') {
      const { imagen } = (await req.json()) as { imagen?: string };
      if (!imagen?.startsWith('data:image/')) return json({ error: 'imagen inválida' }, 400);
      const r = await fal(COLA, { method: 'POST', body: JSON.stringify({ image_url: imagen, resolution: 1024, texture_size: 2048, decimation_target: 200000 }) });
      return json({ id: r.request_id });
    }

    if (url.pathname === '/ia3d/estado') {
      const id = url.searchParams.get('id') ?? '';
      if (!/^[\w-]{8,80}$/.test(id)) return json({ error: 'id inválido' }, 400);
      const e = await fal(`${COLA}/requests/${id}/status`);
      if (e.status !== 'COMPLETED') return json({ estado: e.status, posicion: e.queue_position });
      const r = await fal(`${COLA}/requests/${id}`);
      const glb = (r.model_glb as { url?: string } | undefined)?.url;
      if (!glb) return json({ estado: 'ERROR', error: 'fal.ai no devolvió el modelo' });
      return json({ estado: 'COMPLETED', glb });
    }

    if (url.pathname === '/ia3d/modelo') {
      const origen = url.searchParams.get('url') ?? '';
      // Solo se reenvían archivos de los servidores de fal
      if (!/^https:\/\/([\w-]+\.)*fal\.(media|run|ai)\//.test(origen)) return json({ error: 'URL no permitida' }, 400);
      const r = await fetch(origen);
      if (!r.ok) return json({ error: `no se pudo bajar el modelo (${r.status})` }, 502);
      return new Response(r.body, { headers: { ...CORS, 'Content-Type': 'model/gltf-binary' } });
    }

    return json({ error: 'ruta desconocida' }, 404);
  } catch (e) {
    return json({ error: (e as Error).message }, 502);
  }
}
