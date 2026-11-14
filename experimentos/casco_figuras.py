"""Figuras del informe a partir de casco_paso_a_paso.py (JPEG livianos).

    python experimentos/casco_figuras.py SALIDA/ FIGURAS/
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

plt.rcParams.update({"font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9, "figure.dpi": 130,
                     "axes.spines.top": False, "axes.spines.right": False, "font.family": "DejaVu Sans"})
AZUL, NARANJA, VERDE, ROJO, GRIS = "#1f5fa8", "#d9691c", "#1f8a5b", "#c4262e", "#6b7280"
MARCO = (0.2, 0.15, 0.8, 0.85)


def guardar(fig, ruta, calidad=88):
    fig.savefig(ruta, format="jpeg", pil_kwargs={"quality": calidad}, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("figura", ruta.name, round(ruta.stat().st_size / 1024), "KB")


def main(entrada: str, salida: str) -> None:
    ent, sal = Path(entrada), Path(salida)
    sal.mkdir(parents=True, exist_ok=True)
    V = json.loads((ent / "valores.json").read_text())
    A = np.load(ent / "arreglos.npz")
    foto = A["foto"][:, :, ::-1]
    mascara, disp, Zf, Za, d = A["mascara"], A["disparidad"], A["Z_frente"], A["Z_atras"], A["d"]
    alto, ancho = foto.shape[:2]
    cam = V["camara"]
    ej = {e["nombre"]: e for e in V["ejemplos"]}
    puntos = [("A", ej["visera"]), ("B", ej["casco_arriba"]), ("C", ej["cerca_del_borde"])]

    # 1. Foto de entrada con ejes, recuadro, centro óptico y los tres píxeles de ejemplo
    fig, ax = plt.subplots(figsize=(4.6, 5.9))
    ax.imshow(foto)
    x0, y0, x1, y1 = MARCO[0] * ancho, MARCO[1] * alto, MARCO[2] * ancho, MARCO[3] * alto
    ax.plot([x0, x1, x1, x0, x0], [y0, y0, y1, y1, y0], "--", color="#ffd400", lw=1.4, label="recuadro de encuadre")
    for k in (1, 2):
        ax.plot([ancho * k / 3] * 2, [0, alto], color="white", lw=0.6, alpha=0.6)
        ax.plot([0, ancho], [alto * k / 3] * 2, color="white", lw=0.6, alpha=0.6)
    ax.plot(cam["cx"], cam["cy"], "+", color="#ffd400", ms=14, mew=2)
    ax.annotate("$(c_x, c_y)$", (cam["cx"], cam["cy"]), (cam["cx"] - 250, cam["cy"] + 150), color="#ffd400", fontsize=10,
                arrowprops=dict(arrowstyle="-", color="#ffd400"), bbox=dict(boxstyle="round,pad=0.2", fc="black", ec="none", alpha=0.55))
    for nombre, e in puntos:
        ax.plot(e["u"], e["v"], "o", mfc="none", mec=ROJO, mew=2, ms=10)
        ax.text(e["u"] + 14, e["v"] - 14, nombre, color="white", fontsize=12, weight="bold",
                bbox=dict(boxstyle="round,pad=0.15", fc=ROJO, ec="none"))
    ax.set_xlabel("u (columna, píxeles)")
    ax.set_ylabel("v (fila, píxeles)")
    ax.set_title(f"Foto de entrada reducida: {ancho} × {alto} píxeles")
    ax.legend(loc="lower right", fontsize=7, framealpha=0.9)
    guardar(fig, sal / "casco_entrada.jpg")

    # 2. Etapas: foto, máscara, disparidad, profundidad
    fig, axs = plt.subplots(1, 4, figsize=(11, 4.6))
    dentro = mascara > 0
    axs[0].imshow(foto); axs[0].set_title("Foto  $I$")
    axs[1].imshow(mascara, cmap="gray"); axs[1].set_title("Máscara  $M_{vu}\\in\\{0,1\\}$")
    dm = np.where(dentro, disp, np.nan)
    im = axs[2].imshow(dm, cmap="turbo"); axs[2].set_title("Disparidad  $D(u,v)$")
    fig.colorbar(im, ax=axs[2], fraction=0.046, pad=0.03)
    zm = np.where(dentro, Zf, np.nan)
    im = axs[3].imshow(zm, cmap="viridis_r"); axs[3].set_title("Profundidad  $Z_{frente}$ (mm)")
    fig.colorbar(im, ax=axs[3], fraction=0.046, pad=0.03)
    for a in axs:
        a.set_xticks([]); a.set_yticks([])
    guardar(fig, sal / "casco_etapas.jpg")

    # 3. Perfil a lo largo de la fila v = 576 (visera): disparidad y profundidades
    v = ej["visera"]["v"]
    fila = dentro[v]
    u_obj = np.nonzero(fila)[0]
    u = np.arange(ancho)
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(6.6, 5.2), sharex=True)
    a1.plot(u[u_obj], disp[v][u_obj], color=AZUL, lw=1.4, label="$D(u,v)$ de la red")
    a1.axhline(V["profundidad"]["D_borde"], color=GRIS, ls=":", lw=1); a1.text(u_obj[-1] - 70, V["profundidad"]["D_borde"] - 0.32, "$D_{borde}$", color=GRIS)
    a1.axhline(V["profundidad"]["D_alto"], color=GRIS, ls=":", lw=1); a1.text(u_obj[-1] - 70, V["profundidad"]["D_alto"] + 0.1, "$D_{alto}$", color=GRIS)
    a1.set_ylabel("disparidad (relativa)"); a1.legend(loc="lower right", fontsize=8)
    a1.set_title(f"Fila v = {v}: de la disparidad a la profundidad")
    a2.fill_between(u[u_obj], Zf[v][u_obj], Za[v][u_obj], color="#cfe3f7", label="espesor del sólido")
    a2.plot(u[u_obj], Zf[v][u_obj], color=AZUL, lw=1.6, label="$Z_{frente}$")
    a2.plot(u[u_obj], Za[v][u_obj], color=NARANJA, lw=1.6, label="$Z_{atrás}$")
    a2.axhline(V["profundidad"]["Z0"], color=GRIS, ls="--", lw=1); a2.text(u_obj[0] + 4, V["profundidad"]["Z0"] - 6, "$Z_0$ = 400 mm", color=GRIS)
    for nombre, e in puntos:
        if e["v"] == v:
            a2.plot(e["u"], e["Z_frente"], "o", color=ROJO); a2.annotate(nombre, (e["u"], e["Z_frente"]), (e["u"] + 8, e["Z_frente"] - 22), color=ROJO, weight="bold")
    a2.invert_yaxis(); a2.set_xlabel("u (columna, píxeles)"); a2.set_ylabel("Z (mm)  ← más cerca"); a2.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=3, fontsize=8)
    guardar(fig, sal / "casco_perfil.jpg")

    # 4. Corte lateral (plano Y–Z) en la columna del punto B, con los rayos de la cámara
    uB = ej["casco_arriba"]["u"]
    col = np.nonzero(dentro[:, uB])[0]
    f, cy = cam["f"], cam["cy"]
    Yf = (col - cy) / f * Zf[col, uB]
    Ya = (col - cy) / f * Za[col, uB]
    fig, ax = plt.subplots(figsize=(6.6, 4.4))
    ax.plot(Zf[col, uB], -Yf, color=AZUL, lw=2, label="frente $P=Z\\,K^{-1}[u,v,1]^T$")
    ax.plot(Za[col, uB], -Ya, color=NARANJA, lw=2, label="atrás")
    for k in np.linspace(0, len(col) - 1, 9).astype(int):
        ax.plot([0, Zf[col[k], uB]], [0, -Yf[k]], color="#9aa4b2", lw=0.7)
    ax.plot(0, 0, "ks", ms=7); ax.text(6, 18, "cámara\n(centro óptico)", fontsize=8)
    ax.set_xlabel("Z (mm, hacia adelante)"); ax.set_ylabel("−Y (mm, hacia arriba)")
    ax.set_aspect("equal"); ax.set_title(f"Corte lateral en la columna u = {uB}")
    ax.legend(loc="lower right", fontsize=8); ax.grid(alpha=0.25)
    guardar(fig, sal / "casco_corte.jpg")

    # 5. Histograma de profundidades del frente
    fig, ax = plt.subplots(figsize=(5.4, 3.3))
    ax.hist(Zf[dentro], bins=60, color=AZUL, alpha=0.85)
    ax.axvline(V["profundidad"]["Z0"], color=GRIS, ls="--"); ax.text(V["profundidad"]["Z0"] - 2, ax.get_ylim()[1] * 0.92, "$Z_0$", ha="right", color=GRIS)
    ax.set_xlabel("$Z_{frente}$ (mm)"); ax.set_ylabel("número de píxeles")
    ax.set_title(f"Profundidades del frente ({int(dentro.sum()):,} píxeles)".replace(",", " "))
    guardar(fig, sal / "casco_histograma.jpg")

    # 6. Nube de puntos (3D) del frente
    Vm = A["V"]; nf = int(A["n_frente"])
    idx = np.arange(0, nf, 5)
    fig = plt.figure(figsize=(6.2, 5.2))
    ax = fig.add_subplot(projection="3d")
    sc = ax.scatter(Vm[idx, 0], Vm[idx, 2], Vm[idx, 1], c=Vm[idx, 2], s=1.2, cmap="viridis")
    ax.set_xlabel("x (mm)"); ax.set_ylabel("z (mm)"); ax.set_zlabel("y (mm)")
    ax.set_box_aspect((np.ptp(Vm[:, 0]), np.ptp(Vm[:, 2]), np.ptp(Vm[:, 1])))
    ax.view_init(elev=14, azim=-58); ax.set_title("Vértices del frente en los ejes del objeto")
    guardar(fig, sal / "casco_nube.jpg")

    # 7. Zoom a la malla: triángulos alrededor del triángulo del ejemplo
    T = V["malla"]["triangulo"]
    F = A["F"].astype(np.int64)
    n_f = F.shape[0]
    nfr = V["malla"]["caras"]["frente"]
    Ff = F[:nfr]
    c = np.array(T["V"]).mean(axis=0)
    cerca = np.nonzero((np.abs(Vm[Ff][:, :, 0] - c[0]).max(axis=1) < 4.5) & (np.abs(Vm[Ff][:, :, 1] - c[1]).max(axis=1) < 4.5))[0]
    fig, ax = plt.subplots(figsize=(5.4, 5.0))
    for k in cerca:
        t = Vm[Ff[k]]
        ax.fill(t[:, 0], t[:, 1], fc="#eef3fa", ec="#7d8aa0", lw=0.6)
    t = np.array(T["V"])
    ax.fill(t[:, 0], t[:, 1], fc="#ffd9b8", ec=NARANJA, lw=1.8)
    for k, etq in enumerate(["$p_0$", "$p_1$", "$p_2$"]):
        ax.plot(t[k, 0], t[k, 1], "o", color=NARANJA); ax.annotate(etq, (t[k, 0], t[k, 1]), (t[k, 0] + 0.15, t[k, 1] + 0.15), fontsize=11, color=NARANJA)
    ax.set_aspect("equal"); ax.set_xlabel("x (mm)"); ax.set_ylabel("y (mm)")
    ax.set_title("Malla $F$: un triángulo y sus vecinos (vista de frente)")
    guardar(fig, sal / "casco_malla_zoom.jpg")

    # 8. Tiempos por etapa
    tm = V["tiempos_ms"]
    fig, ax = plt.subplots(figsize=(5.4, 2.9))
    nombres = ["Segmentación\n(U²-Net)", "Profundidad\n(red + fórmula)", "Malla $V$, $F$\n($K^{-1}$ y triángulos)"]
    vals = [tm["segmentacion"], tm["profundidad"], tm["malla"]]
    b = ax.bar(nombres, vals, color=[AZUL, NARANJA, VERDE])
    for r, val in zip(b, vals):
        ax.text(r.get_x() + r.get_width() / 2, val + max(vals) * 0.02, f"{val} ms", ha="center")
    ax.set_ylabel("tiempo (ms)"); ax.set_title(f"Tiempo de cada etapa con la foto del casco (total {tm['total']} ms)"); ax.set_ylim(0, max(vals) * 1.18)
    guardar(fig, sal / "casco_tiempos_etapas.jpg")

    # 9. Diagrama de la tubería con los tamaños de cada objeto
    fig, ax = plt.subplots(figsize=(11, 2.9))
    ax.set_xlim(0, 11); ax.set_ylim(0, 2.9); ax.axis("off")
    cajas = [("Foto $I$", f"{alto}×{ancho}×3\n({alto * ancho * 3:,} números)".replace(",", " ")), ("Máscara $M$", f"{alto}×{ancho}\n(0 ó 1)"),
             ("Profundidad $Z$", f"{alto}×{ancho}\n(mm)"), ("Puntos $3\\times N$", f"N = {V['malla']['n_frente']:,}\nnodos".replace(",", " ")),
             ("Malla $V$, $F$", f"V: {V['malla']['N']:,}×3\nF: {V['malla']['M']:,}×3".replace(",", " "))]
    pasos = ["1. segmentar\nU²-Net", "2. profundidad\nDepth Anything V2", "3. retroproyectar\n$K^{-1}$ y diag(Z)", "4. unir en\ntriángulos"]
    ancho_c, sep = 1.65, 0.62
    x = 0.15
    for i, (t, sub) in enumerate(cajas):
        ax.add_patch(FancyBboxPatch((x, 0.7), ancho_c, 1.5, boxstyle="round,pad=0.04", fc="#eef3fa", ec=AZUL, lw=1.4))
        ax.text(x + ancho_c / 2, 1.8, t, ha="center", va="center", fontsize=10, weight="bold")
        ax.text(x + ancho_c / 2, 1.2, sub, ha="center", va="center", fontsize=8.5, color="#334155")
        if i < len(pasos):
            ax.add_patch(FancyArrowPatch((x + ancho_c + 0.05, 1.45), (x + ancho_c + sep - 0.05, 1.45), arrowstyle="-|>", mutation_scale=14, color=NARANJA, lw=1.6))
            ax.text(x + ancho_c + sep / 2, 0.38, pasos[i], ha="center", va="center", fontsize=7.4, color=NARANJA)
        x += ancho_c + sep
    guardar(fig, sal / "casco_flujo.jpg")

    # 10. Diagrama de hardware: ESP32-CAM → adaptador USB-serial → computador → página
    fig, ax = plt.subplots(figsize=(11, 3.0))
    ax.set_xlim(0, 11); ax.set_ylim(0, 3.0); ax.axis("off")
    bloques = [("Sensor OV2640", "1600×1200 px\npíxel de 2,2 µm", "#fff4d6"), ("ESP32-CAM", "codifica JPEG\n(firmware)", "#e6f6ee"),
               ("Adaptador\nUSB-serial (FTDI)", "UART ↔ USB", "#eef3fa"), ("Computador", "servidor Python\n(FastAPI)", "#f3e8f9"), ("Navegador", "página (three.js)", "#fde8e8")]
    enlaces = ["8 bits\nparalelo", "UART\n1 Mbaud", "cable\nUSB", "HTTP\nlocalhost"]
    ancho_b, sep = 1.62, 0.62
    x = 0.12
    for i, (t, sub, col) in enumerate(bloques):
        ax.add_patch(FancyBboxPatch((x, 0.75), ancho_b, 1.6, boxstyle="round,pad=0.04", fc=col, ec="#475569", lw=1.2))
        ax.text(x + ancho_b / 2, 1.95, t, ha="center", va="center", fontsize=9, weight="bold")
        ax.text(x + ancho_b / 2, 1.3, sub, ha="center", va="center", fontsize=8, color="#334155")
        if i < len(enlaces):
            ax.add_patch(FancyArrowPatch((x + ancho_b + 0.04, 1.55), (x + ancho_b + sep - 0.04, 1.55), arrowstyle="<|-|>" if i == 2 else "-|>", mutation_scale=13, color=NARANJA, lw=1.5))
            ax.text(x + ancho_b + sep / 2, 0.42, enlaces[i], ha="center", va="center", fontsize=7.4, color=NARANJA)
        x += ancho_b + sep
    guardar(fig, sal / "hardware_bloques.jpg")

    # 11. Rotación de los ejes de la cámara a los del objeto (R_o = diag(1,-1,-1))
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.2))
    for a, (tit, ejes) in zip(axs, [("Ejes de la cámara", [("$X$", (1, 0), ROJO), ("$Y$ (abajo)", (0, -1), VERDE)]),
                                      ("Ejes del objeto  $p=R_o(P-c)$", [("$x$", (1, 0), ROJO), ("$y$ (arriba)", (0, 1), VERDE)])]):
        for nombre, (dx, dy), col in ejes:
            a.annotate("", (dx, dy), (0, 0), arrowprops=dict(arrowstyle="-|>", color=col, lw=2))
            a.text(dx * 1.12, dy * 1.12, nombre, color=col, ha="center", va="center")
        a.set_xlim(-1.6, 1.6); a.set_ylim(-1.5, 1.5); a.set_aspect("equal"); a.axis("off"); a.set_title(tit, fontsize=9)
    axs[0].text(0, -1.45, "$Z$ sale hacia adelante (de la hoja hacia ti)", ha="center", fontsize=7.5, color=GRIS)
    axs[1].text(0, -1.45, "$z$ sale hacia la cámara", ha="center", fontsize=7.5, color=GRIS)
    guardar(fig, sal / "casco_rotacion.jpg")

    # 12. DCT de un bloque real del casco
    gris = 0.299 * foto[:, :, 0] + 0.587 * foto[:, :, 1] + 0.114 * foto[:, :, 2]
    mejor, bu, bv = 0, 0, 0
    for vv in range(190, 960, 8):
        for uu in range(200, 744, 8):
            if mascara[vv:vv + 8, uu:uu + 8].all() and gris[vv:vv + 8, uu:uu + 8].var() > mejor:
                mejor, bu, bv = gris[vv:vv + 8, uu:uu + 8].var(), uu, vv
    X = np.round(gris[bv:bv + 8, bu:bu + 8])
    Cm = np.array([[(np.sqrt(1 / 8) if k == 0 else 0.5) * np.cos((2 * j + 1) * k * np.pi / 16) for j in range(8)] for k in range(8)])
    Y = Cm @ (X - 128) @ Cm.T
    Xr = Cm.T @ Y @ Cm + 128
    fig, axs = plt.subplots(1, 3, figsize=(10.5, 3.6))
    for a, M_, tit, cm in zip(axs, [X, np.abs(Y), Xr], ["Bloque $X$ (brillo, 8×8)", "Coeficientes $|Y|=|CXC^T|$", "Recuperado $C^TYC+128$"], ["gray", "magma", "gray"]):
        im = a.imshow(M_, cmap=cm, vmin=0 if cm == "gray" else None, vmax=255 if cm == "gray" else None)
        a.set_title(tit, fontsize=9)
        for (i, j), val in np.ndenumerate(M_):
            a.text(j, i, f"{val:.0f}", ha="center", va="center", fontsize=6.5, color="white" if (cm == "gray" and val < 140) or cm == "magma" else "black")
        a.set_xticks(range(8)); a.set_yticks(range(8)); a.tick_params(labelsize=7)
    fig.tight_layout()
    guardar(fig, sal / "casco_dct.jpg")

    # versión liviana de la foto del casco y de los recortes para el informe
    import PIL.Image as I
    im = I.fromarray(foto)
    im.thumbnail((720, 960)); im.save(sal / "casco_foto.jpg", quality=84)
    rgba = np.dstack([foto, (A["alfa"]).astype(np.uint8)])
    fondo = np.full_like(foto, 255)
    a = (A["alfa"][..., None] / 255.0)
    rec = (foto * a + fondo * (1 - a)).astype(np.uint8)
    vs, us = np.nonzero(A["alfa"] > 127)
    I.fromarray(rec[vs.min():vs.max() + 1, us.min():us.max() + 1]).save(sal / "casco_recorte.jpg", quality=86)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
