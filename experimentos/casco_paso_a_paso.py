"""Paso a paso, con números reales, de la foto de un casco al sólido 3D.

Ejecuta cada etapa del mapeador (servidor/escaner) por separado y guarda los
valores intermedios (matrices, vectores, inversas) para el informe:

    servidor/.venv/bin/python experimentos/casco_paso_a_paso.py FOTO.jpeg SALIDA/

Escribe SALIDA/valores.json y SALIDA/arreglos.npz (foto reducida, máscara,
disparidad, profundidad y malla), que usa experimentos/casco_figuras.py.
"""

import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "servidor"))

from escaner import malla as malla_mod  # noqa: E402
from escaner import profundidad as prof_mod  # noqa: E402
from escaner.geometria import inversa_K, matriz_K  # noqa: E402
from escaner.proceso import ANCHO_PROCESO, MARCO, Parametros, reducir  # noqa: E402
from escaner.segmentacion import segmentar  # noqa: E402


def lista(a, nd=6):
    return np.round(np.asarray(a, dtype=np.float64), nd).tolist()


def main(foto_ruta: str, salida: str) -> None:
    out = Path(salida)
    out.mkdir(parents=True, exist_ok=True)
    p = Parametros()
    V: dict = {}

    original = cv2.imread(foto_ruta, cv2.IMREAD_COLOR)
    V["original"] = {"alto": int(original.shape[0]), "ancho": int(original.shape[1])}
    foto = reducir(original)
    alto, ancho = foto.shape[:2]
    V["foto"] = {"alto": alto, "ancho": ancho, "ancho_proceso": ANCHO_PROCESO,
                 "numeros": alto * ancho * 3, "pixeles": alto * ancho}

    # --- Cámara: K y K⁻¹ ---------------------------------------------------
    K = matriz_K(ancho, alto, p.fov)
    Ki = inversa_K(K)
    V["camara"] = {
        "fov": p.fov, "f": float(K[0, 0]), "cx": float(K[0, 2]), "cy": float(K[1, 2]),
        "K": lista(K), "K_inv": lista(Ki), "K_por_Kinv": lista(K @ Ki, 12),
        "det_K": float(np.linalg.det(K)), "f2": float(K[0, 0] ** 2),
        "tan_medio_fov": float(np.tan(np.radians(p.fov) / 2)),
    }

    # --- 1. Segmentación ----------------------------------------------------
    segmentar(foto, p.marco)          # calentamiento: la primera llamada carga la red
    t = time.perf_counter()
    mascara, alfa, metodo = segmentar(foto, p.marco)
    t_seg = time.perf_counter() - t
    vs, us = np.nonzero(mascara)
    V["segmentacion"] = {
        "metodo": metodo, "area_px": int(mascara.sum()), "fraccion": float(mascara.mean()),
        "u_min": int(us.min()), "u_max": int(us.max()), "v_min": int(vs.min()), "v_max": int(vs.max()),
        "tiempo_ms": round(t_seg * 1000),
    }

    # --- 2. Profundidad -----------------------------------------------------
    gris = cv2.cvtColor(foto, cv2.COLOR_BGR2GRAY)
    prof_mod.disparidad(foto, mascara)  # calentamiento
    t = time.perf_counter()
    disp = prof_mod.disparidad(foto, mascara)
    t_red = time.perf_counter() - t
    mm_por_px = p.distancia_mm / K[0, 0]
    t = time.perf_counter()
    prof = prof_mod.estimar(mascara, gris, p.distancia_mm, mm_por_px, relieve=p.relieve, detalle=p.detalle, disparidad=disp)
    t_prof = time.perf_counter() - t
    d = cv2.distanceTransform(mascara.astype(np.uint8), cv2.DIST_L2, 5)
    d_max = float(d.max())
    dentro = mascara > 0
    V["profundidad"] = {
        "Z0": p.distancia_mm, "mm_por_px": float(mm_por_px), "d_max_px": d_max, "R_mm": float(prof.radio_mm),
        "disparidad_disponible": disp is not None,
        "Zfrente_min": float(prof.Z_frente[dentro].min()), "Zfrente_max": float(prof.Z_frente[dentro].max()),
        "Zatras_min": float(prof.Z_atras[dentro].min()), "Zatras_max": float(prof.Z_atras[dentro].max()),
        "h_max": float(prof.altura[dentro].max()), "tiempo_red_ms": round(t_red * 1000), "tiempo_calculo_ms": round(t_prof * 1000),
    }
    if disp is not None:
        D_borde = float(np.median(disp[dentro & (d <= 2)]))
        D_alto = float(np.percentile(disp[dentro], 99))
        V["profundidad"].update({"D_borde": D_borde, "D_alto": D_alto,
                                 "D_min": float(disp[dentro].min()), "D_max": float(disp[dentro].max())})

    # --- Píxeles de ejemplo (tres puntos del casco) ---------------------------
    ejemplos = {"visera": (488, 576), "casco_arriba": (450, 300), "cerca_del_borde": None}
    # un píxel dentro del objeto, a 6 píxeles del borde, en la fila de la visera
    fila = 576
    candidatos = np.nonzero(dentro[fila] & (d[fila] >= 5.5) & (d[fila] <= 6.5))[0]
    ejemplos["cerca_del_borde"] = (int(candidatos[0]), fila)
    lista_ej = []
    for nombre, (u, v) in ejemplos.items():
        assert dentro[v, u], f"{nombre} ({u},{v}) cae fuera de la máscara"
        s = float(d[v, u] / d_max)
        perfil = float(np.sqrt(max(0.0, 1 - (1 - s) ** 2)))
        e = {"nombre": nombre, "u": int(u), "v": int(v), "d": float(d[v, u]), "s": s, "p_s": perfil,
             "bgr": foto[v, u].tolist()}
        if disp is not None:
            e["D"] = float(disp[v, u])
            e["r"] = float(np.clip((disp[v, u] - D_borde) / max(D_alto - D_borde, 0.15 * (np.percentile(disp, 99) - np.percentile(disp, 1)), 1e-6), 0, 1))
        e["h"] = float(prof.altura[v, u])
        e["Z_frente"] = float(prof.Z_frente[v, u])
        e["Z_atras"] = float(prof.Z_atras[v, u])
        pix = np.array([u, v, 1.0])
        rayo = Ki @ pix
        P = e["Z_frente"] * rayo
        e["pixel_h"] = lista(pix)
        e["rayo"] = lista(rayo)
        e["P"] = lista(P)
        e["K_P"] = lista(K @ P)                       # = Z·[u, v, 1]
        e["K_P_entre_Z"] = lista((K @ P) / P[2])      # recupera [u, v, 1]
        # sistema 2×3 de la proyección para este píxel y su espacio nulo
        A = np.array([[K[0, 0], 0, K[0, 2] - u], [0, K[1, 1], K[1, 2] - v]])
        _, sv, vt = np.linalg.svd(A)
        nulo = vt[-1] / vt[-1][2]
        e["A_2x3"] = lista(A)
        e["rango_A"] = int(np.linalg.matrix_rank(A))
        e["espacio_nulo"] = lista(nulo)
        lista_ej.append(e)
    V["ejemplos"] = lista_ej

    # --- Retroproyección de muchos píxeles a la vez: 3×N·diag(Z) --------------
    vs_n, us_n = np.nonzero(dentro)
    sel = np.linspace(0, len(us_n) - 1, 5).astype(int)
    Hm = np.vstack([us_n[sel], vs_n[sel], np.ones(len(sel))])
    Zs = prof.Z_frente[vs_n[sel], us_n[sel]].astype(np.float64)
    V["lote"] = {"H": lista(Hm), "Z": lista(Zs), "K_inv_H": lista(Ki @ Hm), "P": lista((Ki @ Hm) * Zs)}

    # --- 4. Malla -------------------------------------------------------------
    t = time.perf_counter()
    m = malla_mod.construir(mascara, prof.Z_frente, prof.Z_atras, K, p.distancia_mm, alfa)
    t_malla = time.perf_counter() - t
    Vm, Fm = m.V.astype(np.float64), m.F.astype(np.int64)
    tam = Vm.max(axis=0) - Vm.min(axis=0)
    # un triángulo del frente cerca del centro del casco
    tri = None
    mejor = 1e18
    for k in range(m.caras["frente"]):
        c = Vm[Fm[k]].mean(axis=0)
        dist = (c[0] ** 2 + c[1] ** 2)
        if dist < mejor:
            mejor, tri = dist, k
    t3 = Vm[Fm[tri]]
    e1, e2 = t3[1] - t3[0], t3[2] - t3[0]
    n = np.cross(e1, e2)
    V["malla"] = {
        "N": int(len(Vm)), "M": int(len(Fm)), "n_frente": int(m.n_frente), "caras": m.caras,
        "tam_mm": lista(tam, 2), "centro_camara": lista(m.centro, 3),
        "volumen_mm3": float(malla_mod.volumen(m.V, m.F)), "tiempo_ms": round(t_malla * 1000),
        "R_objeto": lista(malla_mod.R_OBJETO),
        "triangulo": {"indice": int(tri), "F": Fm[tri].tolist(), "V": lista(t3, 4), "e1": lista(e1, 4), "e2": lista(e2, 4),
                      "normal": lista(n, 4), "normal_unitaria": lista(n / np.linalg.norm(n), 6),
                      "det": float(np.linalg.det(t3)), "volumen_tetraedro": float(np.linalg.det(t3) / 6)},
        "ejemplo_V": lista(Vm[:4], 4), "ejemplo_F": Fm[:4].tolist(),
    }
    # Tiempos de cada etapa con el mismo código que usa el servidor (segunda corrida, con las redes ya cargadas)
    from escaner.proceso import escanear  # noqa: E402

    escanear(original, p)
    t2 = escanear(original, p).tiempos
    V["tiempos_ms"] = {"segmentacion": t2["segmentacion"], "profundidad": t2["profundidad"], "malla": t2["malla"],
                       "total": t2["segmentacion"] + t2["profundidad"] + t2["malla"]}

    (out / "valores.json").write_text(json.dumps(V, ensure_ascii=False, indent=1))
    np.savez_compressed(
        out / "arreglos.npz", foto=foto, mascara=mascara, alfa=alfa,
        disparidad=(disp if disp is not None else np.zeros((alto, ancho), np.float32)),
        d=d.astype(np.float32), Z_frente=prof.Z_frente, Z_atras=prof.Z_atras, h=prof.altura,
        V=m.V, F=m.F, uv=m.uv, n_frente=m.n_frente,
    )
    print(json.dumps({k: V[k] for k in ("foto", "camara", "segmentacion", "profundidad")}, ensure_ascii=False, indent=1)[:2500])


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
