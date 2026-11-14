"""Figuras de la habitación y de los vectores del proceso (JPEG livianos).

    python experimentos/habitacion_figuras.py HAB_SALIDA/ CASCO_SALIDA/ FIGURAS/

HAB_SALIDA: salida de habitacion_paso_a_paso.py; CASCO_SALIDA: la de casco_paso_a_paso.py.
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402
from mpl_toolkits.mplot3d import proj3d  # noqa: E402,F401

plt.rcParams.update({"font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9, "figure.dpi": 130,
                     "axes.spines.top": False, "axes.spines.right": False, "font.family": "DejaVu Sans"})
AZUL, NARANJA, VERDE, ROJO, GRIS, MORADO = "#1f5fa8", "#d9691c", "#1f8a5b", "#c4262e", "#6b7280", "#7a3fb0"


def guardar(fig, ruta, calidad=88):
    fig.savefig(ruta, format="jpeg", pil_kwargs={"quality": calidad}, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("figura", ruta.name, round(ruta.stat().st_size / 1024), "KB")


def flecha3d(ax, o, v, color, etiqueta=None, lw=2, desp=(0, 0, 0), fs=10):
    o, v = np.asarray(o, float), np.asarray(v, float)
    ax.quiver(*o, *v, color=color, lw=lw, arrow_length_ratio=0.12)
    if etiqueta:
        p = o + v + np.asarray(desp)
        ax.text(*p, etiqueta, color=color, fontsize=fs)


def flecha2d(ax, a, b, color, lw=1.8, estilo="-|>", **kw):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle=estilo, mutation_scale=12, color=color, lw=lw, **kw))


def main(ent_h: str, ent_c: str, salida: str) -> None:
    eh, ec, sal = Path(ent_h), Path(ent_c), Path(salida)
    sal.mkdir(parents=True, exist_ok=True)
    V = json.loads((eh / "hab_valores.json").read_text())
    A = np.load(eh / "hab_arreglos.npz")
    C = json.loads((ec / "valores.json").read_text())
    AC = np.load(ec / "arreglos.npz")
    foto = A["foto"][:, :, ::-1]
    Z, Zp, L, Zv, Z2 = A["Z"], A["Zp"], A["L"], A["Zv"], A["Z2"]
    ep = A["etiqueta_plano"]
    alto, ancho = foto.shape[:2]
    cam, ej, planos = V["camara"], V["ejemplos"], V["planos"]
    pn = {p["nombre"]: p for p in planos}

    # ---------------------------------------------------------------- 1. entrada
    fig, ax = plt.subplots(figsize=(7.2, 4.9))
    ax.imshow(foto)
    ax.plot(cam["cx"], cam["cy"], "+", color="#ffd400", ms=14, mew=2)
    ax.annotate("$(c_x, c_y)$", (cam["cx"], cam["cy"]), (cam["cx"] + 40, cam["cy"] - 90), color="#ffd400", fontsize=10,
                arrowprops=dict(arrowstyle="-", color="#ffd400"), bbox=dict(boxstyle="round,pad=0.2", fc="black", ec="none", alpha=0.55))
    for nom, e in (("piso", ej["piso"]), ("pared", ej["pared"]), ("techo", ej["techo"]), ("mueble", ej["mueble"])):
        ax.plot(e["u"], e["v"], "o", mfc="none", mec=ROJO, mew=2, ms=11)
        ax.text(e["u"] + 14, e["v"] - 12, nom, color="white", fontsize=10, weight="bold",
                bbox=dict(boxstyle="round,pad=0.15", fc=ROJO, ec="none"))
    ax.set_xlabel("u (columna, píxeles)")
    ax.set_ylabel("v (fila, píxeles)")
    ax.set_title(f"Foto de la sala reducida: {ancho} × {alto} píxeles, con los cuatro píxeles de ejemplo")
    guardar(fig, sal / "hab_entrada.jpg")

    # ---------------------------------------------------------------- 2. etapas
    fig, axs = plt.subplots(2, 2, figsize=(9.6, 6.6))
    axs[0, 0].imshow(foto); axs[0, 0].set_title("Foto  $I$")
    im = axs[0, 1].imshow(Z, cmap="turbo"); axs[0, 1].set_title("Profundidad de la red  $Z(u,v)$ [m]")
    fig.colorbar(im, ax=axs[0, 1], fraction=0.046, pad=0.02)
    cls = np.zeros(L.shape, np.int32)   # 0 mueble, 1 piso, 2 techo, 3 pared, 4 ventana/espejo
    cls[L == 3], cls[L == 5], cls[L == 0] = 1, 2, 3
    cls[(L == 8) | (L == 27) | (L == 14)] = 4
    cmap = ListedColormap(["#b08a5a", "#6aa84f", "#f1c232", "#9fc5e8", "#cc4125"])
    axs[1, 0].imshow(cls, cmap=cmap, vmin=0, vmax=4, interpolation="nearest")
    axs[1, 0].set_title("Clase de cada píxel (SegFormer)")
    for k, (nom, col) in enumerate((("mueble u otro", "#b08a5a"), ("piso", "#6aa84f"), ("techo", "#f1c232"), ("pared", "#9fc5e8"), ("ventana/puerta", "#cc4125"))):
        axs[1, 0].plot([], [], "s", color=col, label=nom)
    axs[1, 0].legend(loc="lower center", fontsize=7, ncol=3, framealpha=0.9)
    cm2 = plt.get_cmap("tab10")
    mapa = np.ones((alto, ancho, 3))
    for j in range(len(planos)):
        mapa[ep == j] = cm2(j)[:3]
    mapa[ep < 0] = foto[ep < 0] / 255 * 0.35 + 0.55
    axs[1, 1].imshow(mapa); axs[1, 1].set_title("Planos ajustados (un color por plano)")
    for j, p in enumerate(planos):
        axs[1, 1].plot([], [], "s", color=cm2(j), label=p["nombre"])
    axs[1, 1].legend(loc="lower center", fontsize=7, ncol=3, framealpha=0.9)
    for a in axs.ravel():
        a.set_xticks([]); a.set_yticks([])
    fig.tight_layout()
    guardar(fig, sal / "hab_etapas.jpg")

    # ---------------------------------------------------------------- 3. RANSAC: tres puntos, e1, e2, n
    R = V["ransac_ejemplo"]
    Pa = np.array(R["puntos"])
    e1, e2, nn = np.array(R["e1"]), np.array(R["e2"]), np.array(R["cruz"])
    Pp = A["Pp"]
    fig = plt.figure(figsize=(9.6, 4.4))
    ax = fig.add_subplot(1, 2, 1, projection="3d")
    ys, xs = np.nonzero(L == 3)
    sel = np.random.default_rng(1).choice(len(xs), 1500, replace=False)
    Pf = Pp[ys[sel], xs[sel]]
    ax.scatter(Pf[:, 0], Pf[:, 2], -Pf[:, 1], s=2, c="#6aa84f", alpha=0.5)
    a0 = Pa[0]
    flecha3d(ax, [a0[0], a0[2], -a0[1]], [e1[0], e1[2], -e1[1]], AZUL, "$\\vec e_1$", desp=(0, 0, 0.05))
    flecha3d(ax, [a0[0], a0[2], -a0[1]], [e2[0], e2[2], -e2[1]], NARANJA, "$\\vec e_2$", desp=(0, 0, 0.05))
    nv = np.array(R["n"]) * 0.35
    flecha3d(ax, [a0[0], a0[2], -a0[1]], [nv[0], nv[2], -nv[1]], ROJO, "$\\hat n$", desp=(0, 0, 0.05))
    for k, (nombre, P_) in enumerate(zip("abc", Pa)):
        ax.scatter([P_[0]], [P_[2]], [-P_[1]], s=40, c=ROJO, depthshade=False)
        ax.text(P_[0], P_[2], -P_[1] - 0.12, f"$P_{k}$", fontsize=10)
    ax.set_zlim(-1.3, -0.4)
    ax.set_xlabel("x [m]"); ax.set_ylabel("z (hacia el fondo) [m]"); ax.set_zlabel("altura  $-y$ [m]")
    ax.set_title("Tres puntos del piso, $\\vec e_1$, $\\vec e_2$ y la normal")
    ax.view_init(elev=22, azim=-62)
    ax2 = fig.add_subplot(1, 2, 2)
    dist = np.abs(Pf @ np.array(R["n"]) - R["d"])
    ax2.hist(dist * 100, bins=40, color=AZUL, alpha=0.85)
    ax2.axvline(R["tolerancia"] * 100, color=ROJO, ls="--", lw=1.6)
    ax2.text(R["tolerancia"] * 100 + 0.2, ax2.get_ylim()[1] * 0.85, f"tolerancia\n{R['tolerancia'] * 100:.1f} cm", color=ROJO, fontsize=8)
    ax2.set_xlabel("distancia de cada punto al plano  $|\\hat n\\cdot P - d|$ [cm]")
    ax2.set_ylabel("puntos del piso (muestra)")
    ax2.set_title(f"Votos del plano: {R['votos']:,} de {R['de']:,}".replace(",", " "))
    fig.tight_layout()
    guardar(fig, sal / "hab_ransac.jpg")

    # ---------------------------------------------------------------- 4. rayo-plano (vista lateral, plano y-z)
    pe = ej["piso"]
    n = np.array(pn["piso"]["n"]); d = pn["piso"]["d"]
    r = np.array(pe["rayo"])
    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    zz = np.linspace(0, 6, 50)
    # plano: n_y*y + n_z*z = d  (x = 0 corte)
    yy = (d - n[2] * zz) / n[1]
    ax.plot(zz, -yy, color=VERDE, lw=2.2, label="plano del piso  $\\vec n\\cdot P=d$")
    ax.fill_between(zz, -yy, -yy - 0.25, color=VERDE, alpha=0.12)
    ax.plot(0, 0, "ko", ms=7); ax.text(-0.15, 0.2, "cámara  $O$", fontsize=9)
    tf = pe["Z_plano"]; tr = pe["Z_red"]
    ax.plot([0, 5.2 * 1], [0, -5.2 * r[1]], color=GRIS, lw=1, ls=":")
    flecha2d(ax, (0, 0), (1.0, -r[1] * 1.0), AZUL, lw=2.4)
    ax.text(1.3, -r[1] * 1.3 + 0.12, "$\\vec r=K^{-1}\\vec p$", color=AZUL, fontsize=11)
    ax.plot([tr], [-r[1] * tr], "o", color=NARANJA, ms=9); ax.annotate(f"punto de la red\n$Z={tr:.2f}$ m", (tr, -r[1] * tr), (tr + 0.35, -r[1] * tr + 0.9), color=NARANJA, fontsize=9, arrowprops=dict(arrowstyle="-", color=NARANJA))
    ax.plot([tf], [-r[1] * tf], "o", color=VERDE, ms=9); ax.annotate(f"en el plano\n$Z=d/(\\vec n\\cdot\\vec r)={tf:.2f}$ m", (tf, -r[1] * tf), (tf - 0.1, -r[1] * tf - 0.8), color=VERDE, fontsize=9, arrowprops=dict(arrowstyle="-", color=VERDE))
    # normal desde el punto del piso (hacia arriba)
    base = np.array([tf, -r[1] * tf])
    flecha2d(ax, base, base + np.array([-n[2], -n[1]]) * 0.9, ROJO, lw=2.2)
    ax.text(base[0] - 0.9, base[1] + 0.95, "$\\vec n$", color=ROJO, fontsize=11)
    ax.annotate("", (0, 0), (0, -(-d / n[1]) * -1), arrowprops=dict(arrowstyle="<->", color=MORADO, lw=1.6)) if False else None
    ax.plot([0, 0], [0, d / n[1] * -1 * -1 * -1], color=MORADO, lw=0)  # reserva
    ax.annotate("", xy=(0.0, -(d / n[1])), xytext=(0.0, 0.0), arrowprops=dict(arrowstyle="<->", color=MORADO, lw=1.6))
    ax.text(0.08, -(d / n[1]) / 2, f"$|d|\\approx{abs(d):.2f}$ m\n(altura de la cámara)", color=MORADO, fontsize=9)
    ax.set_xlim(-0.4, 5.9); ax.set_ylim(-2.4, 0.9)
    ax.set_xlabel("$z$ (hacia el fondo de la sala) [m]"); ax.set_ylabel("$-y$ (altura) [m]")
    ax.set_title("Corte lateral: el rayo del píxel corta el plano del piso")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_aspect("equal")
    guardar(fig, sal / "hab_rayo_plano.jpg")

    # ---------------------------------------------------------------- 5. planos: vista desde arriba y de lado
    fig, axs = plt.subplots(1, 2, figsize=(10, 4.6))
    rng = np.random.default_rng(2)
    for j, p in enumerate(planos):
        ys, xs = np.nonzero(ep == j)
        if not len(xs):
            continue
        s = rng.choice(len(xs), min(1800, len(xs)), replace=False)
        Q = Pp[ys[s], xs[s]]
        axs[0].scatter(Q[:, 0], Q[:, 2], s=2, color=cm2(j), label=p["nombre"])
        axs[1].scatter(Q[:, 2], -Q[:, 1], s=2, color=cm2(j))
    for a in axs:
        a.plot(0, 0, "k^" if a is axs[0] else "ko", ms=8)
    axs[0].text(0.1, 0.1, "cámara", fontsize=9)
    axs[0].set_xlabel("$x$ [m]"); axs[0].set_ylabel("$z$ [m]"); axs[0].set_title("Vista desde arriba (plano $x$–$z$)")
    axs[0].set_aspect("equal"); axs[0].legend(fontsize=7, markerscale=4, loc="upper left")
    axs[1].set_xlabel("$z$ [m]"); axs[1].set_ylabel("$-y$ (altura) [m]"); axs[1].set_title("Vista de lado (plano $z$–$y$)")
    axs[1].set_aspect("equal")
    fig.tight_layout()
    guardar(fig, sal / "hab_planos.jpg")

    # ---------------------------------------------------------------- 6. cuarto vacío detrás del sofá
    pm = ej["mueble"]
    fig, axs = plt.subplots(1, 2, figsize=(10, 4.2))
    nombres = [c["plano"] for c in pm["cortes"]]
    ts = [c["t"] for c in pm["cortes"]]
    col = []
    mejor = min(t for t in ts if t is not None and t > 0.1)
    for t in ts:
        col.append(VERDE if t == mejor else (GRIS if (t is not None and t > 0) else "#e5e7eb"))
    vals = [t if (t is not None and 0 < t < 15) else 0 for t in ts]
    b = axs[0].barh(nombres, vals, color=col)
    for rect, t in zip(b, ts):
        txt = f"{t:.2f} m" if t > 0 else f"{t:.2f} (negativo: detrás de la cámara)"
        axs[0].text(max(rect.get_width(), 0) + 0.15, rect.get_y() + rect.get_height() / 2, txt, va="center", fontsize=8)
    axs[0].invert_yaxis(); axs[0].set_xlim(0, 11)
    axs[0].set_xlabel("$t_k=d_k/(\\vec n_k\\cdot\\vec r)$ [m]")
    axs[0].set_title("Cortes del rayo del píxel del sofá con cada plano")
    # vista desde arriba: cámara, rayo, sofá, pared
    rx = pm["rayo"][0]
    zS, zW = pm["Z_red"], mejor
    axs[1].plot([0, rx * zW * 1.05], [0, zW * 1.05], color=AZUL, lw=2)
    axs[1].plot(0, 0, "k^", ms=9); axs[1].text(0.05, 0.08, "cámara", fontsize=9)
    axs[1].plot([rx * zS], [zS], "o", color=NARANJA, ms=10); axs[1].text(rx * zS + 0.1, zS, f"sofá  $Z={zS:.2f}$ m", color=NARANJA, fontsize=9)
    axs[1].plot([rx * zW], [zW], "o", color=VERDE, ms=10); axs[1].text(rx * zW + 0.1, zW - 0.1, f"pared de atrás (cuarto vacío)\n$Z={zW:.2f}$ m", color=VERDE, fontsize=9)
    axs[1].set_aspect("equal"); axs[1].set_xlabel("$x$ [m]"); axs[1].set_ylabel("$z$ [m]")
    axs[1].set_title("Vista desde arriba: lo que hay detrás del sofá")
    fig.tight_layout()
    guardar(fig, sal / "hab_vacio.jpg")

    # ---------------------------------------------------------------- 7. salto y corte de triángulos
    tj = V["triangulos_ejemplo"]
    v0 = tj["salto"]["nodos_uv"][0][1]
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.9), gridspec_kw={"width_ratios": [1.5, 1]})
    axs[0].plot(np.arange(ancho), Z2[v0], color=AZUL, lw=1.4, label="$Z$ después de los planos")
    axs[0].plot(np.arange(ancho), Z[v0], color=GRIS, lw=0.9, alpha=0.8, label="$Z$ de la red")
    u0 = tj["salto"]["nodos_uv"][0][0]
    axs[0].axvline(u0, color=ROJO, ls="--", lw=1.4)
    axs[0].text(u0 + 6, Z2[v0].max() * 0.92, "salto: la malla\nse corta aquí", color=ROJO, fontsize=9)
    axs[0].set_xlabel("u (columna)"); axs[0].set_ylabel("$Z$ [m]")
    axs[0].set_title(f"Fila $v={v0}$: del sofá a la pared del fondo"); axs[0].legend(fontsize=8, loc="lower right")
    tp, ts_ = tj["plano"], tj["salto"]
    etq = ["triángulo del piso", "triángulo del salto"]
    cosv = [tp["coseno"], ts_["coseno"]]
    bb = axs[1].bar(etq, cosv, color=[VERDE, ROJO])
    sen = np.sin(np.radians(V["rejilla"]["canto_grados"]))
    axs[1].axhline(sen, color="k", ls="--", lw=1.2); axs[1].text(0.5, sen + 0.015, "sen 5° = %.3f" % sen, ha="center", fontsize=8)
    for rect, t in zip(bb, cosv):
        axs[1].text(rect.get_x() + rect.get_width() / 2, t + 0.01, f"{t:.3f}", ha="center", fontsize=9)
    axs[1].set_ylabel("$|\\hat n\\cdot\\hat r|$"); axs[1].set_title("Producto punto normal–rayo")
    fig.tight_layout()
    guardar(fig, sal / "hab_corte.jpg")

    # ---------------------------------------------------------------- 8. rotación de Rodrigues
    rot = V["rotacion"]
    PP = A["PP"]; uu, vv = A["uu"], A["vv"]
    G = np.array(rot["G"])
    sel = (vv > 0.65 * alto) & ~(L[vv, uu] > 0) | ((vv > 0.65 * alto) & (L[vv, uu] == 3))
    ptos = PP[(vv > 0.65 * alto) & (L[vv, uu] == 3)]
    ptos = ptos[np.random.default_rng(3).choice(len(ptos), 4000, replace=False)]
    Vr = ptos @ G.T
    fig, axs = plt.subplots(1, 3, figsize=(11, 3.8))
    axs[0].scatter(ptos[:, 2], -ptos[:, 1], s=2, c=AZUL)
    axs[0].set_title("Piso antes de girar (ejes de la cámara)"); axs[0].set_xlabel("$z$ [mm]"); axs[0].set_ylabel("$-y$ [mm]")
    axs[1].scatter(Vr[:, 2], Vr[:, 1], s=2, c=VERDE)
    axs[1].set_title("Piso después de girar con $G=R_oR$"); axs[1].set_xlabel("$z$ [mm]"); axs[1].set_ylabel("$y$ [mm]")
    axs[2].hist(ptos[:, 1], bins=50, color=AZUL, alpha=0.6, label=f"antes: $\\sigma={rot['altura_y_antes_std_mm']:.0f}$ mm")
    axs[2].hist(Vr[:, 1], bins=50, color=VERDE, alpha=0.7, label=f"después: $\\sigma={rot['altura_y_despues_std_mm']:.1f}$ mm")
    axs[2].set_xlabel("coordenada vertical de los puntos del piso [mm]"); axs[2].set_ylabel("puntos"); axs[2].legend(fontsize=8)
    axs[2].set_title("Dispersión de la altura del piso")
    fig.tight_layout()
    guardar(fig, sal / "hab_rotacion.jpg")

    # ---------------------------------------------------------------- 9. error por plano
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.8))
    nom = [p["nombre"] for p in planos]
    x = np.arange(len(nom))
    axs[0].bar(x - 0.2, [p["disp_antes_m"] * 100 for p in planos], 0.4, color=ROJO, label="puntos de la red")
    axs[0].bar(x + 0.2, [max(p["disp_despues_m"] * 100, 1e-5) for p in planos], 0.4, color=VERDE, label="sobre el plano")
    axs[0].set_yscale("log"); axs[0].set_xticks(x); axs[0].set_xticklabels(nom, rotation=25, fontsize=8)
    axs[0].set_ylabel("desviación respecto al plano [cm]"); axs[0].legend(fontsize=8)
    axs[0].set_title("Planitud de cada superficie")
    axs[1].bar(x, [p["error_Z_antes_pct"] for p in planos], color=AZUL)
    axs[1].set_xticks(x); axs[1].set_xticklabels(nom, rotation=25, fontsize=8)
    axs[1].set_ylabel("cambio medio de $Z$ al usar el plano [%]"); axs[1].set_title("Cuánto corrige el plano a la red")
    fig.tight_layout()
    guardar(fig, sal / "hab_planitud.jpg")

    # ---------------------------------------------------------------- 10. nube de puntos coloreada, dos vistas
    FF_ = A["FF"]
    col_ = foto[vv, uu] / 255
    Q = PP @ G.T
    sel = np.random.default_rng(4).choice(len(Q), 100000, replace=False)
    fig = plt.figure(figsize=(10, 4.4))
    for k, (el, az, t) in enumerate(((10, -90, "Desde el punto de la foto"), (28, -35, "Girada: se ve la profundidad"))):
        ax = fig.add_subplot(1, 2, k + 1, projection="3d")
        ax.scatter(Q[sel, 0], -Q[sel, 2], Q[sel, 1], s=1.6, c=col_[sel])
        ax.view_init(elev=el, azim=az)
        ax.set_title(t)
        ax.set_xlabel("$x$ [mm]"); ax.set_ylabel("profundidad [mm]"); ax.set_zlabel("$y$ [mm]")
        ax.set_box_aspect((1, 1.4, 0.8))
        ax.tick_params(labelsize=6)
    fig.tight_layout()
    guardar(fig, sal / "hab_nube.jpg")

    # ---------------------------------------------------------------- 11. normales de los planos como vectores y sus ángulos
    fig = plt.figure(figsize=(9.2, 4.2))
    ax = fig.add_subplot(1, 2, 1, projection="3d")
    colores = {"piso": VERDE, "techo": "#b58900", "pared 1": AZUL, "pared 3": ROJO}
    for nom_, c_ in colores.items():
        nv = np.array(pn[nom_]["n"])
        flecha3d(ax, [0, 0, 0], [nv[0], nv[2], -nv[1]], c_, f"$\\hat n_{{\\mathrm{{{nom_.replace(' ', '')}}}}}$", lw=2.2, desp=(0, 0, 0.05), fs=9)
    ax.set_xlim(-1, 1); ax.set_ylim(-0.4, 1); ax.set_zlim(-1, 1)
    ax.set_xlabel("$x$"); ax.set_ylabel("$z$"); ax.set_zlabel("arriba")
    ax.set_title("Normales unitarias de los planos"); ax.view_init(elev=20, azim=-55)
    ax2 = fig.add_subplot(1, 2, 2)
    ks = ["piso", "techo", "pared 1", "pared 3"]
    M = np.array([[np.array(pn[a]["n"]) @ np.array(pn[b]["n"]) for b in ks] for a in ks])
    ang = np.degrees(np.arccos(np.clip(np.abs(M), 0, 1)))
    im = ax2.imshow(ang, cmap="YlGnBu", vmin=0, vmax=90)
    for i in range(4):
        for j in range(4):
            ax2.text(j, i, f"{ang[i, j]:.1f}°", ha="center", va="center", fontsize=9, color="black" if ang[i, j] < 60 else "white")
    ax2.set_xticks(range(4)); ax2.set_xticklabels(ks, rotation=25, fontsize=8); ax2.set_yticks(range(4)); ax2.set_yticklabels(ks, fontsize=8)
    ax2.set_title("Ángulo entre planos (de $\\hat n_i\\cdot\\hat n_j$)")
    fig.colorbar(im, ax=ax2, fraction=0.046, pad=0.03)
    fig.tight_layout()
    guardar(fig, sal / "hab_normales.jpg")

    # ---------------------------------------------------------------- 12. flujo de la habitación
    fig, ax = plt.subplots(figsize=(10.4, 2.6)); ax.axis("off"); ax.set_xlim(0, 10.4); ax.set_ylim(0, 2.6)
    cajas = [("1. Red", "$Z(u,v)$ [m]\nclases"), ("2. Planos", "RANSAC:\n$\\hat n_k,\\ d_k$"), ("3. Rayo–plano", "$Z=d/(\\hat n\\cdot\\vec r)$"),
             ("4. Malla", "$V\\in\\mathbb{R}^{N\\times3}$\n$F$"), ("5. Atrás", "cuarto vacío\n+ LaMa"), ("6. Enderezar", "$G=R_oR$\n(Rodrigues)")]
    for i, (t, s) in enumerate(cajas):
        x0 = 0.1 + i * 1.7
        ax.add_patch(FancyBboxPatch((x0, 0.5), 1.45, 1.5, boxstyle="round,pad=0.03", fc="#eaf1fb", ec=AZUL, lw=1.4))
        ax.text(x0 + 0.725, 1.65, t, ha="center", fontsize=9, weight="bold", color=AZUL)
        ax.text(x0 + 0.725, 1.05, s, ha="center", va="center", fontsize=8)
        if i < len(cajas) - 1:
            flecha2d(ax, (x0 + 1.48, 1.25), (x0 + 1.7, 1.25), GRIS)
    ax.text(5.2, 0.15, "foto  →  matrices  $K^{-1}$, $H$  →  planos  →  vértices y triángulos  →  modelo girado", ha="center", fontsize=9, color=GRIS)
    guardar(fig, sal / "hab_flujo.jpg")

    # =============================================================== VECTORES DEL CASCO
    ejc = {e["nombre"]: e for e in C["ejemplos"]}
    a, b_, c_ = ejc["visera"], ejc["casco_arriba"], ejc["cerca_del_borde"]
    camc = C["camara"]
    fc, cxc, cyc = camc["f"], camc["cx"], camc["cy"]
    # ---- V1: del píxel al rayo y al punto (3D)
    fig = plt.figure(figsize=(9.4, 4.6))
    ax = fig.add_subplot(1, 2, 1, projection="3d")
    W_, H_ = C["foto"]["ancho"], C["foto"]["alto"]
    # plano de imagen a z=1 (coordenadas normalizadas)
    x0, x1 = (0 - cxc) / fc, (W_ - cxc) / fc
    y0, y1 = (0 - cyc) / fc, (H_ - cyc) / fc
    ax.plot_surface(*np.meshgrid([x0, x1], [y0, y1])[0:1], np.ones((2, 2)), alpha=0) if False else None
    xs_ = np.array([[x0, x1], [x0, x1]]); ys_ = np.array([[y0, y0], [y1, y1]])
    ax.plot_surface(xs_, np.ones((2, 2)), -ys_, alpha=0.12, color=AZUL)
    ax.scatter([0], [0], [0], c="k", s=40)
    ax.text(0, 0, 0.08, "$O$", fontsize=10)
    flecha3d(ax, [0, 0, 0], [0, 1.3, 0], GRIS, "eje óptico", lw=1.4, desp=(0, 0, 0.05), fs=8)
    for nom, e, col in (("A", a, ROJO), ("B", b_, NARANJA), ("C", c_, VERDE)):
        rr = np.array(e["rayo"])
        flecha3d(ax, [0, 0, 0], [rr[0], 1.0, -rr[1]], col, f"$\\vec r_{nom}$", lw=2, desp=(0, 0.02, 0.04), fs=10)
        Pn = np.array(e["P"]) / e["Z_frente"] * 1.9
        ax.plot([0, Pn[0]], [0, 1.9], [0, -Pn[1]], color=col, ls=":", lw=1)
        ax.scatter([Pn[0]], [1.9], [-Pn[1]], c=col, s=22, depthshade=False)
    ax.set_xlabel("$x$"); ax.set_ylabel("$z$"); ax.set_zlabel("$-y$")
    ax.set_title("Rayos $\\vec r=K^{-1}\\vec p$ (con $r_z=1$) y sus puntos")
    ax.view_init(elev=18, azim=-68)
    ax2 = fig.add_subplot(1, 2, 2)
    # ángulos con el eje óptico y entre rayos
    rA, rB, rC = (np.array(e["rayo"]) for e in (a, b_, c_))
    def ang_(x, y):
        return np.degrees(np.arccos(np.clip(x @ y / np.linalg.norm(x) / np.linalg.norm(y), -1, 1)))
    ejeZ = np.array([0, 0, 1.0])
    etq = ["A–eje", "B–eje", "C–eje", "A–B", "A–C", "B–C"]
    vals = [ang_(rA, ejeZ), ang_(rB, ejeZ), ang_(rC, ejeZ), ang_(rA, rB), ang_(rA, rC), ang_(rB, rC)]
    cols = [AZUL] * 3 + [MORADO] * 3
    bars = ax2.bar(etq, vals, color=cols)
    for rect, t in zip(bars, vals):
        ax2.text(rect.get_x() + rect.get_width() / 2, t + 0.3, f"{t:.1f}°", ha="center", fontsize=8)
    ax2.set_ylabel("ángulo $\\theta$  (de $\\cos\\theta=\\vec a\\cdot\\vec b/\\|\\vec a\\|\\|\\vec b\\|$)")
    ax2.set_title("Ángulos entre rayos (producto punto)")
    fig.tight_layout()
    guardar(fig, sal / "vec_rayos.jpg")

    # ---- V2: color como vector en R^3
    foto_c = AC["foto"]
    def rgb_(e):
        return np.array([int(e["bgr"][2]), int(e["bgr"][1]), int(e["bgr"][0])], float)
    # colores reales leídos de la foto en la posición (u,v)
    def leer(e):
        bgr = foto_c[e["v"], e["u"]]
        return np.array([bgr[2], bgr[1], bgr[0]], float)
    cA, cB, cC = leer(a), leer(b_), leer(c_)
    fig = plt.figure(figsize=(9.4, 4.4))
    ax = fig.add_subplot(1, 2, 1, projection="3d")
    for nom, v_, col in (("A", cA, ROJO), ("B", cB, NARANJA), ("C", cC, VERDE)):
        flecha3d(ax, [0, 0, 0], v_, col, f"$\\vec c_{nom}$", lw=2.2, desp=(4, 4, 6), fs=10)
    flecha3d(ax, cA + np.array([0, 0, 8]), cB - cA, MORADO, "$\\vec c_B-\\vec c_A$", lw=1.8, desp=(-70, 0, -25), fs=9)
    ax.set_xlim(0, 255); ax.set_ylim(0, 255); ax.set_zlim(0, 255)
    ax.set_xlabel("R"); ax.set_ylabel("G"); ax.set_zlabel("B")
    ax.set_title("Colores de tres píxeles como vectores de $\\mathbb{R}^3$")
    ax.view_init(elev=22, azim=-50)
    ax2 = fig.add_subplot(1, 2, 2)
    pares = {"A–B": (cA, cB), "A–C": (cA, cC), "B–C": (cB, cC)}
    dist = [np.linalg.norm(x - y) for x, y in pares.values()]
    cosn = [x @ y / np.linalg.norm(x) / np.linalg.norm(y) if np.linalg.norm(x) * np.linalg.norm(y) > 0 else 1 for x, y in pares.values()]
    x_ = np.arange(3)
    ax2.bar(x_ - 0.2, dist, 0.4, color=MORADO, label="distancia $\\|\\vec c_i-\\vec c_j\\|$")
    for xi, t in zip(x_, dist):
        ax2.text(xi - 0.2, t + 4, f"{t:.1f}", ha="center", fontsize=8)
    ax2.set_xticks(x_); ax2.set_xticklabels(list(pares)); ax2.set_ylabel("norma de la diferencia")
    ax3 = ax2.twinx(); ax3.spines["right"].set_visible(True)
    ax3.bar(x_ + 0.2, cosn, 0.4, color=AZUL, label="coseno del ángulo"); ax3.set_ylim(0, 1.15); ax3.set_ylabel("$\\cos\\theta$")
    for xi, t in zip(x_, cosn):
        ax3.text(xi + 0.2, t + 0.02, f"{t:.3f}", ha="center", fontsize=8, color=AZUL)
    ax2.set_title("Norma y producto punto entre colores")
    h1, l1 = ax2.get_legend_handles_labels(); h2, l2 = ax3.get_legend_handles_labels()
    ax2.legend(h1 + h2, l1 + l2, fontsize=7, loc="upper center")
    fig.tight_layout()
    guardar(fig, sal / "vec_colores.jpg")
    np.save(Path(ent_h).parent / "colores_vec.npy", np.stack([cA, cB, cC]))

    # ---- V3: el triángulo del casco: e1, e2, n = e1 × e2, n̂, y el rayo
    tri = C["malla"]["triangulo"]
    Vt = np.array(tri["V"]); e1c, e2c = np.array(tri["e1"]), np.array(tri["e2"])
    ncr, nu = np.array(tri["normal"]), np.array(tri["normal_unitaria"])
    fig = plt.figure(figsize=(9.4, 4.4))
    ax = fig.add_subplot(1, 2, 1, projection="3d")
    P0 = Vt[0]
    tri_pts = np.vstack([Vt, Vt[0]])
    ax.plot(tri_pts[:, 0], tri_pts[:, 1], tri_pts[:, 2], color=GRIS, lw=1.5)
    ax.plot_trisurf(Vt[:, 0], Vt[:, 1], Vt[:, 2], color="#dbeafe", alpha=0.7)
    flecha3d(ax, P0, e1c, AZUL, "$\\vec e_1$", desp=(0, 0, 0.1))
    flecha3d(ax, P0, e2c, NARANJA, "$\\vec e_2$", desp=(0.1, 0, 0))
    cen = Vt.mean(axis=0)
    flecha3d(ax, cen, nu * 0.25, ROJO, "$\\hat n$", lw=2.4, desp=(0.05, 0, 0.1))
    ax.set_xlabel("$x$ [mm]"); ax.set_ylabel("$y$ [mm]"); ax.set_zlabel("$z$ [mm]")
    ax.set_title("Triángulo de la malla del casco")
    ax.view_init(elev=24, azim=-60)
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.axis("off")
    lineas = [
        ("$\\vec e_1=p_1-p_0$", f"({e1c[0]:.4f}, {e1c[1]:.4f}, {e1c[2]:.4f})"),
        ("$\\vec e_2=p_2-p_0$", f"({e2c[0]:.4f}, {e2c[1]:.4f}, {e2c[2]:.4f})"),
        ("$\\vec n=\\vec e_1\\times\\vec e_2$", f"({ncr[0]:.4f}, {ncr[1]:.4f}, {ncr[2]:.4f})"),
        ("$\\|\\vec n\\|$", f"{np.linalg.norm(ncr):.4f}"),
        ("$\\hat n=\\vec n/\\|\\vec n\\|$", f"({nu[0]:.4f}, {nu[1]:.4f}, {nu[2]:.4f})"),
        ("$\\hat n\\cdot\\vec e_1$", f"{nu @ e1c:.1e}  (≈ 0: perpendicular)"),
        ("$\\hat n\\cdot\\vec e_2$", f"{nu @ e2c:.1e}  (≈ 0: perpendicular)"),
        ("$\\frac{1}{2}\\|\\vec n\\|$ = área", f"{0.5 * np.linalg.norm(ncr):.4f} mm²"),
    ]
    for i, (k, s) in enumerate(lineas):
        ax2.text(0.0, 0.95 - i * 0.115, k, fontsize=10, color=AZUL, va="top")
        ax2.text(0.42, 0.95 - i * 0.115, s, fontsize=9, va="top")
    ax2.set_title("Operaciones con vectores sobre un triángulo")
    fig.tight_layout()
    guardar(fig, sal / "vec_triangulo.jpg")

    # ---- V4: Rodrigues, vectores n, m, v
    n_ = np.array(rot["n"]); m_ = np.array(rot["m"]); v_ = np.array(rot["v"])
    fig = plt.figure(figsize=(9.0, 4.0))
    ax = fig.add_subplot(1, 2, 1, projection="3d")
    flecha3d(ax, [0, 0, 0], [n_[0], n_[2], -n_[1]], ROJO, "$\\hat n$ (normal del piso medida)", lw=2.4, desp=(0, 0, 0.06), fs=8)
    flecha3d(ax, [0, 0, 0], [m_[0], m_[2], -m_[1]], VERDE, "$\\vec m$ (vertical)", lw=2.4, desp=(0, 0, -0.12), fs=8)
    nv_ = np.linalg.norm(v_)
    flecha3d(ax, [0, 0, 0], [v_[0] / nv_, v_[2] / nv_, -v_[1] / nv_], MORADO, "$\\vec v=\\hat n\\times\\vec m$", lw=2, desp=(0.05, 0, 0), fs=8)
    ax.set_xlim(-1, 1); ax.set_ylim(-1, 1); ax.set_zlim(-0.2, 1.1)
    ax.set_xlabel("$x$"); ax.set_ylabel("$z$"); ax.set_zlabel("arriba")
    ax.set_title(f"Giro de {rot['angulo']:.2f}°"); ax.view_init(elev=18, azim=-40)
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.axis("off")
    for i, (k, s) in enumerate([
        ("$\\vec v=\\hat n\\times\\vec m$", f"({v_[0]:.5f}, {v_[1]:.5f}, {v_[2]:.5f})"),
        ("$c=\\hat n\\cdot\\vec m$", f"{rot['c']:.6f}"),
        ("$\\theta=\\arccos c$", f"{rot['angulo']:.3f}°"),
        ("$R\\,\\hat n$", f"({rot['R_n'][0]:+.4f}, {rot['R_n'][1]:+.4f}, {rot['R_n'][2]:+.4f}) = $\\vec m$"),
        ("$\\det R$", f"{rot['det_R']:.6f}"),
    ]):
        ax2.text(0.0, 0.92 - i * 0.16, k, fontsize=10, color=AZUL, va="top")
        ax2.text(0.38, 0.92 - i * 0.16, s, fontsize=9, va="top")
    ax2.set_title("Vectores de la rotación de Rodrigues")
    fig.tight_layout()
    guardar(fig, sal / "vec_rodrigues.jpg")


if __name__ == "__main__":
    main(*sys.argv[1:4])
