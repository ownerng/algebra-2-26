"""Paso a paso, con números reales, de la foto de una habitación a su malla 3D.

Corre las etapas de servidor/escaner/habitacion.py por separado y guarda los
valores intermedios (planos, rayos, intersecciones, rotación) para el informe:

    servidor/.venv/bin/python experimentos/habitacion_paso_a_paso.py FOTO.jpg SALIDA/

Escribe SALIDA/hab_valores.json y SALIDA/hab_arreglos.npz, que usan
experimentos/habitacion_figuras.py y experimentos/habitacion_latex.py.
"""

import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "servidor"))

from escaner import calibracion  # noqa: E402
from escaner import habitacion as hab  # noqa: E402
from escaner import profundidad as prof  # noqa: E402
from escaner.geometria import inversa_K, retroproyectar  # noqa: E402
from escaner.proceso import Parametros, reducir  # noqa: E402

NOMBRES = {0: "pared", 3: "piso", 5: "techo", 8: "ventana", 14: "puerta", 27: "espejo"}


def lista(a, nd=6):
    return np.round(np.asarray(a, dtype=np.float64), nd).tolist()


def main(ruta: str, salida: str) -> None:
    out = Path(salida)
    out.mkdir(parents=True, exist_ok=True)
    p = Parametros()
    V: dict = {}
    original = cv2.imread(ruta, cv2.IMREAD_COLOR)
    foto = reducir(original)
    alto, ancho = foto.shape[:2]
    V["foto"] = {"alto": alto, "ancho": ancho, "original": list(original.shape[:2]), "pixeles": alto * ancho}

    K = calibracion.K(ancho, alto, p.fov)
    Ki = inversa_K(K)
    f = float(K[0, 0])
    V["camara"] = {"fov": p.fov, "f": f, "cx": float(K[0, 2]), "cy": float(K[1, 2]), "K": lista(K), "K_inv": lista(Ki),
                   "det_K": float(np.linalg.det(K))}

    # --- paso 1: profundidad en metros y clases ------------------------------
    prof.metros(foto)
    prof.etiquetas(foto)   # calentamiento
    t = time.perf_counter()
    Z = prof.metros(foto)
    t_prof = time.perf_counter() - t
    t = time.perf_counter()
    L = prof.etiquetas(foto)
    t_seg = time.perf_counter() - t
    clases, cuenta = np.unique(L, return_counts=True)
    V["profundidad"] = {"Zmin": float(Z.min()), "Zmax": float(Z.max()), "Zmedia": float(Z.mean()), "ms": round(t_prof * 1000)}
    V["clases"] = {"ms": round(t_seg * 1000), "n_clases": int(len(clases)),
                   "estructura": {NOMBRES[c]: int(n) for c, n in zip(clases, cuenta) if int(c) in NOMBRES},
                   "frac": {NOMBRES[c]: float(n / L.size) for c, n in zip(clases, cuenta) if int(c) in NOMBRES}}

    # --- rayos y puntos de la red (antes de los planos) ------------------------
    u, v = np.meshgrid(np.arange(ancho), np.arange(alto))
    r = (Ki @ np.stack([u.ravel(), v.ravel(), np.ones(u.size)])).T        # N × 3, r_z = 1
    P = r * Z.ravel()[:, None].astype(np.float64)

    # --- paso 2: planos (RANSAC didáctico: la primera terna del piso) ----------
    rng = np.random.default_rng(0)
    idx_piso = np.flatnonzero(L.ravel() == 3)
    sub = idx_piso[rng.choice(len(idx_piso), min(len(idx_piso), 20000), replace=False)]
    Ps = P[sub]
    ar = np.random.default_rng(7)
    while True:
        tri = ar.integers(0, len(Ps), 3)
        a, b, c = Ps[tri]
        e1, e2 = b - a, c - a
        nn = np.cross(e1, e2)
        if np.linalg.norm(nn) / (np.linalg.norm(e1) * np.linalg.norm(e2)) > 0.3:
            break
    nn_u = nn / np.linalg.norm(nn)
    nn_u *= -1.0 if nn_u @ hab.ARRIBA < 0 else 1.0
    d0 = float(nn_u @ a)
    tol = 0.02 * float(np.median(Ps[:, 2]))
    V["ransac_ejemplo"] = {
        "indices": [int(sub[i]) for i in tri],
        "pix": [[int(sub[i] % ancho), int(sub[i] // ancho)] for i in tri],
        "puntos": lista([a, b, c], 3), "e1": lista(e1, 3), "e2": lista(e2, 3), "cruz": lista(nn, 3),
        "norma_cruz": float(np.linalg.norm(nn)), "n": lista(nn_u, 4), "d": d0,
        "tolerancia": tol, "votos": int((np.abs(Ps @ nn_u - d0) < tol).sum()), "de": len(Ps),
    }

    # --- paso 2-3: maqueta instrumentada --------------------------------------
    rng = np.random.default_rng(0)
    z = Z.ravel().astype(np.float64)
    plano_z, vacio = z.copy(), np.full(z.size, np.inf)
    planos = []
    etiqueta_plano = np.full(z.size, -1, np.int32)
    for clases_e, veces in hab.ESTRUCTURA:
        resto = np.flatnonzero(np.isin(L.ravel(), clases_e))
        for _ in range(veces):
            if len(resto) < 0.01 * z.size:
                break
            sub = resto[rng.choice(len(resto), min(len(resto), 20000), replace=False)]
            pl = hab.ransac(P[sub], rng, minimo=0.15)
            if pl is None:
                break
            n, d, dentro = pl
            with np.errstate(divide="ignore", invalid="ignore"):
                tt = d / (r @ n)
            vacio = np.where(tt > 0.1, np.minimum(vacio, tt), vacio)
            cerca = (tt[resto] > 0) & (np.abs(np.log(np.maximum(tt[resto], 1e-6) / z[resto])) < 0.1)
            sel = resto[cerca]
            antes = np.abs(P[sel] @ n - d)
            plano_z[sel] = tt[sel]
            etiqueta_plano[sel] = len(planos)
            nombre = {(3,): "piso", (5,): "techo"}.get(clases_e, "pared")
            planos.append({"nombre": nombre, "n": lista(n, 5), "d": float(d), "n_pix": int(len(sel)),
                           "frac_clase": float(len(sel) / max(1, (L == clases_e[0]).sum())),
                           "disp_antes_m": float(antes.std()),
                           "angulo_vertical": float(np.degrees(np.arccos(np.clip(n @ hab.ARRIBA, -1, 1))))})
            resto = resto[~cerca]
    # nombres de paredes en orden
    k = 0
    for pl in planos:
        if pl["nombre"] == "pared":
            k += 1
            pl["nombre"] = f"pared {k}"
    Zp = plano_z.reshape(alto, ancho).astype(np.float32)
    Zv = vacio.reshape(alto, ancho)
    V["planos"] = planos
    for pl in planos:
        n = np.array(pl["n"])
        sel = np.flatnonzero(etiqueta_plano == planos.index(pl))
        despues = np.abs(retroproyectar(Ki, u.ravel()[sel], v.ravel()[sel], plano_z[sel]) @ n - pl["d"])
        pl["disp_despues_m"] = float(despues.std())
        pl["error_Z_antes_pct"] = float(np.mean(np.abs(z[sel] - plano_z[sel]) / plano_z[sel]) * 100)

    # --- píxeles de ejemplo: uno del piso, uno de la pared, uno detrás del sofá -
    def pixel(u0, v0):
        i = v0 * ancho + u0
        pix = np.array([u0, v0, 1.0])
        ray = Ki @ pix
        info = {"u": u0, "v": v0, "clase": NOMBRES.get(int(L[v0, u0]), f"clase {int(L[v0, u0])}"), "rayo": lista(ray, 5),
                "norma_rayo": float(np.linalg.norm(ray)), "Z_red": float(Z[v0, u0]), "Z_plano": float(Zp[v0, u0]),
                "P_red": lista(ray * Z[v0, u0], 4), "P_plano": lista(ray * Zp[v0, u0], 4),
                "angulo_eje": float(np.degrees(np.arctan(np.hypot(ray[0], ray[1])))),
                "cortes": [], "vacio": float(Zv[v0, u0]) if np.isfinite(Zv[v0, u0]) else None}
        for j, pl in enumerate(planos):
            n = np.array(pl["n"])
            nr = float(n @ ray)
            info["cortes"].append({"plano": pl["nombre"], "n_dot_r": nr, "t": float(pl["d"] / nr) if abs(nr) > 1e-9 else None})
        return info

    clase_pix = {}
    for nombre, c in (("piso", 3), ("techo", 5), ("pared", 0)):
        ys, xs = np.nonzero(L == c)
        clase_pix[nombre] = (xs, ys)
    ej = {}
    # el píxel del piso más cercano a un punto elegido (abajo-centro)
    for nombre, objetivo in (("piso", (0.62, 0.86)), ("pared", (0.16, 0.45)), ("techo", (0.45, 0.04))):
        xs, ys = clase_pix[nombre]
        ob = np.array([objetivo[0] * ancho, objetivo[1] * alto])
        j = int(np.argmin((xs - ob[0]) ** 2 + (ys - ob[1]) ** 2))
        ej[nombre] = pixel(int(xs[j]), int(ys[j]))
    # detrás del mueble: el píxel del sofá o la mesa con vacío finito
    muebles_clases = ~np.isin(L, [c for cl, _ in hab.ESTRUCTURA for c in cl])
    ys, xs = np.nonzero(muebles_clases & np.isfinite(Zv) & (Zv > Z * 1.12))
    ob = np.array([0.18 * ancho, 0.72 * alto])
    j = int(np.argmin((xs - ob[0]) ** 2 + (ys - ob[1]) ** 2))
    ej["mueble"] = pixel(int(xs[j]), int(ys[j]))
    V["ejemplos"] = ej

    # --- paso 3: mejora del piso con ruido (los puntos del piso a cada plano) ----
    Pp = retroproyectar(Ki, u.ravel(), v.ravel(), plano_z)
    # --- paso 4: malla --------------------------------------------------------
    Z2, _ = hab.maqueta(Z, K, L, np.random.default_rng(0))
    Z2 = hab.alinear(Z2, L, f)
    para_lama, detras, Z_fondo = hab.capa_oculta(Z2, f)
    mueble = ~np.isin(L, [c for cl, _ in hab.ESTRUCTURA for c in cl])
    paso = max(1, int(np.ceil(max(ancho, alto) / hab.LADO)))
    uu, vv, PP, FF = hab.rejilla(Z2, K, paso, mueble=mueble)
    # cuántos triángulos habría sin el corte
    U, Vg = np.meshgrid(np.arange(0, ancho, paso), np.arange(0, alto, paso))
    nx, ny = U.shape[1], U.shape[0]
    total_sin_corte = 2 * (nx - 1) * (ny - 1)
    V["rejilla"] = {"paso": paso, "nx": nx, "ny": ny, "nodos": int(len(PP)), "triangulos_sin_corte": total_sin_corte,
                    "triangulos": int(len(FF)), "cortados": total_sin_corte - int(len(FF)),
                    "canto_grados": hab.CANTO, "corte": hab.CORTE}
    # dos triángulos de ejemplo: uno plano (piso) y uno cortado (salto sofá/pared)
    indice = np.arange(len(uu))
    a_, b_ = np.arange(0, 1), None
    # reconstruir todos los triángulos para buscar ejemplos
    dentro_idx = np.arange(nx * ny).reshape(ny, nx)
    A, B = dentro_idx[:-1, :-1].ravel(), dentro_idx[:-1, 1:].ravel()
    D, E = dentro_idx[1:, :-1].ravel(), dentro_idx[1:, 1:].ravel()
    Fall = np.concatenate([np.stack([A, D, B], 1), np.stack([B, D, E], 1)])
    Pall = retroproyectar(Ki, U.ravel(), Vg.ravel(), Z2[Vg.ravel(), U.ravel()] * 1000)
    tt3 = Pall[Fall]
    n3 = np.cross(tt3[:, 1] - tt3[:, 0], tt3[:, 2] - tt3[:, 0])
    r3 = tt3.sum(axis=1)
    cos3 = np.abs((n3 * r3).sum(axis=1)) / np.maximum(np.linalg.norm(n3, axis=1) * np.linalg.norm(r3, axis=1), 1e-12)
    raz3 = tt3[:, :, 2].max(axis=1) / tt3[:, :, 2].min(axis=1)
    salto3 = (cos3 < np.sin(np.radians(hab.CANTO))) & (raz3 > 1 + hab.CORTE)

    def ejemplo_tri(i):
        tr = tt3[i]
        return {"nodos_uv": [[int(U.ravel()[k]), int(Vg.ravel()[k])] for k in Fall[i]], "vertices": lista(tr, 1),
                "e1": lista(tr[1] - tr[0], 1), "e2": lista(tr[2] - tr[0], 1), "n": lista(n3[i], 0),
                "norma_n": float(np.linalg.norm(n3[i])), "centro": lista(r3[i], 1), "norma_c": float(np.linalg.norm(r3[i])),
                "punto": float(abs(n3[i] @ r3[i])), "coseno": float(cos3[i]), "razon_Z": float(raz3[i]),
                "se_corta": bool(salto3[i])}

    # triángulo plano: del piso, centro de la foto abajo
    candidatos = np.flatnonzero(~salto3 & (cos3 > 0.3) & (raz3 < 1.03))
    cc = tt3[candidatos].mean(axis=1)
    c_ = np.array([0.6 * ancho, 0.85 * alto])
    pix_c = np.array([[(Ki @ [1, 1, 1])[0]]])  # no se usa
    # elegir por cercanía en píxel al punto objetivo
    ctr_uv = np.array([[U.ravel()[Fall[i]].mean(), Vg.ravel()[Fall[i]].mean()] for i in candidatos[:: max(1, len(candidatos) // 4000)]])
    cand_sub = candidatos[:: max(1, len(candidatos) // 4000)]
    ip = cand_sub[int(np.argmin(((ctr_uv - c_) ** 2).sum(axis=1)))]
    ic = np.flatnonzero(salto3)
    ctr_uv2 = np.array([[U.ravel()[Fall[i]].mean(), Vg.ravel()[Fall[i]].mean()] for i in ic])
    # salto entre sofá y pared: cercano al centro-izquierda
    ic0 = ic[int(np.argmin(((ctr_uv2 - np.array([0.40 * ancho, 0.55 * alto])) ** 2).sum(axis=1)))]
    V["triangulos_ejemplo"] = {"plano": ejemplo_tri(int(ip)), "salto": ejemplo_tri(int(ic0))}

    # --- paso 6: rotación de Rodrigues -----------------------------------------
    m, vista_, altura = hab.construir(Z2, K, Z_fondo, detras, mueble)
    piso_pl = next(pl for pl in planos if pl["nombre"] == "piso")
    nP = np.array(piso_pl["n"])
    # el plano del piso con los puntos de la malla (como lo hace construir)
    abajo = vv > 0.65 * alto
    libre = abajo & ~mueble[vv, uu]
    abajo = libre if libre.sum() > 0.05 * abajo.sum() else abajo
    pl_malla = hab.piso(PP[abajo], np.random.default_rng(0))
    n_m, d_m = pl_malla
    M = hab.ARRIBA
    vv_ = np.cross(n_m, M)
    c_dot = float(n_m @ M)
    Cx = hab.cruz(vv_)
    R = np.eye(3) + Cx + Cx @ Cx / (1 + c_dot)
    G = hab.R_OBJETO @ R
    V["rotacion"] = {"n": lista(n_m, 5), "d_mm": float(d_m), "m": lista(M, 1), "v": lista(vv_, 5), "c": c_dot,
                     "angulo": float(np.degrees(np.arccos(np.clip(c_dot, -1, 1)))), "vx": lista(Cx, 5),
                     "vx2": lista(Cx @ Cx, 5), "R": lista(R, 5), "R_n": lista(R @ n_m, 5), "det_R": float(np.linalg.det(R)),
                     "RtR": lista(R.T @ R, 8), "R_OBJETO": lista(hab.R_OBJETO, 1), "G": lista(G, 5),
                     "altura_camara_m": float(altura / 1000)}
    # dispersión de la altura del piso antes y después de girar (puntos del plano)
    Vg_ = PP @ G.T
    sel_piso = hab.ransac(PP[abajo], np.random.default_rng(0))[2]
    pts = PP[abajo][sel_piso]
    V["rotacion"]["altura_y_antes_std_mm"] = float(pts[:, 1].std())
    V["rotacion"]["altura_y_despues_std_mm"] = float((pts @ G.T)[:, 1].std())
    V["rotacion"]["altura_y_despues_media_mm"] = float((pts @ G.T)[:, 1].mean())
    V["malla"] = {"N": int(len(m.V)), "n_frente": int(m.n_frente), "M": int(len(m.F)),
                  "tam_mm": lista(m.V[:m.n_frente].max(axis=0) - m.V[:m.n_frente].min(axis=0), 0),
                  "atras": int(len(m.V) - m.n_frente)}
    # Euler de la capa de adelante no aplica (borde); sí se reporta el número de aristas por la malla completa
    t = time.perf_counter()
    hab.reconstruir(foto)
    V["tiempo_total_s"] = float(time.perf_counter() - t)

    # --- ángulos entre rayos y vectores de la escena ---------------------------
    def ang(a, b):
        return float(np.degrees(np.arccos(np.clip(a @ b / np.linalg.norm(a) / np.linalg.norm(b), -1, 1))))

    rp, rpa = np.array(ej["piso"]["rayo"]), np.array(ej["pared"]["rayo"])
    V["angulos"] = {"piso_pared": ang(rp, rpa), "piso_eje": ej["piso"]["angulo_eje"], "pared_eje": ej["pared"]["angulo_eje"],
                    "dot_piso_pared": float(rp @ rpa)}
    # ángulo entre planos del cuarto (producto punto de normales)
    pw = [pl for pl in planos if pl["nombre"].startswith("pared")]
    pares = {}
    for a_pl in pw[:2]:
        pares[a_pl["nombre"]] = {"con_piso": ang(np.array(a_pl["n"]), nP)}
    if len(pw) >= 2:
        pares["pared1_pared2"] = ang(np.array(pw[0]["n"]), np.array(pw[1]["n"]))
    V["angulos"]["planos"] = pares

    np.savez_compressed(out / "hab_arreglos.npz", foto=foto, Z=Z, Zp=Zp, L=L, Zv=np.where(np.isfinite(Zv), Zv, 0).astype(np.float32),
                        Z2=Z2, etiqueta_plano=etiqueta_plano.reshape(alto, ancho), U=U, Vg=Vg, PP=PP.astype(np.float32),
                        FF=FF.astype(np.int32), uu=uu, vv=vv, Pp=Pp.astype(np.float32).reshape(alto, ancho, 3), G=G)
    (out / "hab_valores.json").write_text(json.dumps(V, indent=1, ensure_ascii=False))
    print(json.dumps({k: V[k] for k in ("foto", "camara", "profundidad", "clases", "rejilla", "malla", "rotacion", "angulos")}, indent=1)[:6000])
    for pl in planos:
        print(pl)
    print(json.dumps(V["ransac_ejemplo"], indent=1))
    print(json.dumps(ej, indent=1)[:5000])
    print(json.dumps(V["triangulos_ejemplo"], indent=1)[:3500])


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
