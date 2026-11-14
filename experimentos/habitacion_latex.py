"""Genera las secciones LaTeX de la sala y de los vectores del casco.

    python experimentos/habitacion_latex.py HAB_SALIDA/ CASCO_SALIDA/ docs/secciones/

Lee hab_valores.json / hab_arreglos.npz (habitacion_paso_a_paso.py), el
valores.json del casco y colores_vec.npy (habitacion_figuras.py). Escribe
vectores_casco.tex, habitacion_matematica.tex y habitacion_resultados.tex.
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from casco_latex import N, fill, n, nm, pm, pmi, sci, vec  # noqa: E402


def r_(x, d):
    return float(np.round(x, d))


def cruz_tex(a, b, d, nombres=("e_1", "e_2")):
    """Las tres componentes del producto cruz a × b con los números (de las cifras mostradas)."""
    a, b = [r_(x, d) for x in a], [r_(x, d) for x in b]
    c = [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
    f = lambda x: n(x, d)  # noqa: E731
    fm = lambda x: f(x) if x >= 0 else rf"\left({f(x)}\right)"  # noqa: E731
    dd = d + 1
    g = lambda x: nm(x, 1 if abs(x) >= 100 else dd)  # noqa: E731
    filas = [
        rf"n_x&=e_{{1y}}e_{{2z}}-e_{{1z}}e_{{2y}}={fm(a[1])}\cdot{fm(b[2])}-{fm(a[2])}\cdot{fm(b[1])}={g(c[0])},",
        rf"n_y&=e_{{1z}}e_{{2x}}-e_{{1x}}e_{{2z}}={fm(a[2])}\cdot{fm(b[0])}-{fm(a[0])}\cdot{fm(b[2])}={g(c[1])},",
        rf"n_z&=e_{{1x}}e_{{2y}}-e_{{1y}}e_{{2x}}={fm(a[0])}\cdot{fm(b[1])}-{fm(a[1])}\cdot{fm(b[0])}={g(c[2])},",
    ]
    return "\\begin{align*}\n" + "\\\\\n".join(filas) + "\n\\end{align*}", np.array(c)


def ang(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return float(np.degrees(np.arccos(np.clip(a @ b / np.linalg.norm(a) / np.linalg.norm(b), -1, 1))))


# =============================================================================
def vectores_casco(C, cols):
    cam = C["camara"]
    K, Ki = np.array(cam["K"]), np.array(cam["K_inv"])
    f, cx, cy = cam["f"], cam["cx"], cam["cy"]
    ej = {e["nombre"]: e for e in C["ejemplos"]}
    puntos = [("A", ej["visera"]), ("B", ej["casco_arriba"]), ("C", ej["cerca_del_borde"])]
    mal = C["malla"]
    tri = mal["triangulo"]
    cA, cB, cC = cols
    rA, rB, rC = (np.array(e["rayo"]) for _, e in puntos)
    # tabla de rayos
    filas = []
    for nom, e in puntos:
        r = np.array(e["rayo"])
        filas.append(rf"{nom} & $({e['u']},\,{e['v']},\,1)$ & $({n(r[0], 4)},\,{n(r[1], 4)},\,1)$ & ${n(np.linalg.norm(r), 4)}$ & ${n(np.degrees(np.arctan(np.hypot(r[0], r[1]))), 2)}^\circ$ & ${n(e['Z_frente'], 1)}$ \\")
    # producto punto A·C paso a paso
    dot = float(rA @ rC)
    nA, nC = float(np.linalg.norm(rA)), float(np.linalg.norm(rC))
    e1, e2 = np.array(tri["e1"]), np.array(tri["e2"])
    cruz_bloque, nvec = cruz_tex(e1, e2, 4)
    nn = float(np.linalg.norm(nvec))
    nu = nvec / nn
    Vt = np.array(tri["V"])
    cen = Vt.sum(axis=0)
    cosv = abs(float(nu @ cen)) / float(np.linalg.norm(cen))
    # colores
    dAB = cB - cA
    dAC = cC - cA
    cos_AB = float(cA @ cB / np.linalg.norm(cA) / np.linalg.norm(cB))
    cos_AC = float(cA @ cC / np.linalg.norm(cA) / np.linalg.norm(cC))
    # del punto de la cámara al objeto: P - c y giro
    c0 = np.array(mal["centro_camara"])
    Ro = np.diag([1.0, -1.0, -1.0])
    filas_obj = []
    for nom, e in puntos:
        P = np.array(e["P"])
        filas_obj.append(rf"{nom} & $({n(P[0],1)},\,{n(P[1],1)},\,{n(P[2],1)})$ & $({n((P - c0)[0],1)},\,{n((P - c0)[1],1)},\,{n((P - c0)[2],1)})$ & $({n((Ro @ (P - c0))[0],1)},\,{n((Ro @ (P - c0))[1],1)},\,{n((Ro @ (P - c0))[2],1)})$ \\")
    plantilla = r"""
\subsection{Los vectores del proceso, uno por uno (casco)}\label{sec:vectores-casco}

Las matrices del proceso se aplican siempre sobre \emph{vectores}, y es con los vectores con los que se hacen las tres operaciones
del curso: la norma $\lVert\vect{a}\rVert$, el producto punto $\vect{a}\cdot\vect{b}$ y el producto cruz $\vect{a}\times\vect{b}$.
Esta subsección sigue, con los números del casco, cada vector que aparece desde el píxel hasta la malla y dice qué significa cada operación.

\paragraph{Del píxel al rayo.} Un píxel es el vector $\vect{p}=(u,v,1)^{T}$ (coordenadas homogéneas). Al multiplicarlo por $K^{-1}$ se obtiene
el vector $\vect{r}=K^{-1}\vect{p}=((u-c_x)/f,\ (v-c_y)/f,\ 1)^{T}$, que apunta desde la cámara hacia el punto del mundo que vio ese píxel. Para
el píxel A, $(u,v)=(@uA@,@vA@)$, por ejemplo,
\begin{equation}
\vect{r}_A=\begin{pmatrix}\dfrac{@uA@-@cx@}{@f@}\\[2mm]\dfrac{@vA@-@cy@}{@f@}\\[1mm]1\end{pmatrix}=@rA@ .
\end{equation}
La \emph{norma} de $\vect{r}$ es la longitud de ese vector, $\lVert\vect{r}_A\rVert=\sqrt{(@rAx@)^2+(@rAy@)^2+1^2}=@nA@$, y el ángulo
que forma con el eje óptico $\vect{e}_z=(0,0,1)^{T}$ sale del producto punto:
\begin{equation}
\cos\theta_A=\frac{\vect{r}_A\cdot\vect{e}_z}{\lVert\vect{r}_A\rVert\,\lVert\vect{e}_z\rVert}=\frac{1}{@nA@}=@cosAz@,\qquad \theta_A=@thA@^\circ .
\end{equation}
Cuanto más lejos del centro está el píxel, más inclinado es su rayo (cuadro~\ref{tab:rayos-casco} y figura~\ref{fig:vec-rayos}).

\begin{table}[htbp]
\caption{Los tres píxeles del casco como vectores: píxel $\vect{p}$, rayo $\vect{r}=K^{-1}\vect{p}$, su norma, su ángulo con el eje óptico y la profundidad $Z$ (mm)}
\label{tab:rayos-casco}
\centering\small
\begin{tabular}{@{}clcccc@{}}
\toprule
Píxel & $\vect{p}=(u,v,1)$ & $\vect{r}=K^{-1}\vect{p}$ & $\lVert\vect{r}\rVert$ & $\theta$ & $Z$ \\
\midrule
@filas_rayos@
\bottomrule
\end{tabular}
\end{table}

\paragraph{Del rayo al punto.} Multiplicar el rayo por la profundidad lo estira hasta el punto del casco: $\vect{P}=Z\,\vect{r}$. Es una
multiplicación de un vector por un escalar, y el resultado conserva la dirección. Para A, con $Z=@ZA@$ mm,
\begin{equation}
\vect{P}_A=@ZA@\cdot@rA@=@PA@\ \text{mm},
\end{equation}
y su norma, $\lVert\vect{P}_A\rVert=@nPA@$ mm, es la distancia real de la cámara a ese punto (mayor que $Z$ porque el rayo está inclinado:
$\lVert\vect{P}\rVert=Z\lVert\vect{r}\rVert$).

\paragraph{Producto punto entre dos rayos.} El ángulo entre los rayos de dos píxeles se calcula igual. Para A y C,
\begin{equation}
\vect{r}_A\cdot\vect{r}_C=(@rAx@)(@rCx@)+(@rAy@)(@rCy@)+1\cdot1=@dotAC@,
\end{equation}
\begin{equation}
\cos\theta_{AC}=\frac{@dotAC@}{(@nA@)(@nC@)}=@cosAC@,
\end{equation}
es decir $\theta_{AC}=@thAC@^\circ$. En la figura~\ref{fig:vec-rayos} están los ángulos de los tres píxeles con el eje óptico y entre sí.

\begin{figure}[htbp]
\centering
\includegraphics[width=0.97\linewidth]{vec_rayos.jpg}
\caption{Izquierda: los rayos $\vect{r}_A$, $\vect{r}_B$ y $\vect{r}_C$ (con componente $z=1$) cruzan el plano de la imagen, y sus puntos
$\vect{P}=Z\vect{r}$ están más lejos sobre el mismo rayo. Derecha: ángulos de cada rayo con el eje óptico y entre rayos, calculados con el
producto punto.}
\label{fig:vec-rayos}
\end{figure}

\paragraph{El color como vector.} El color de un píxel también es un vector, de $\mathbb{R}^{3}$: $\vect{c}=(R,G,B)^{T}$. En la foto, los
píxeles A, B y C tienen $\vect{c}_A=@cAv@$, $\vect{c}_B=@cBv@$ y $\vect{c}_C=@cCv@$. La resta de dos colores es un vector y su norma
mide qué tan distintos son:
\begin{equation}
\vect{c}_B-\vect{c}_A=@dAB@,\qquad \lVert\vect{c}_B-\vect{c}_A\rVert=\sqrt{@d1@^2+@d2@^2+@d3@^2}=@nAB@ .
\end{equation}
El producto punto da otra información: $\cos\theta=\vect{c}_A\cdot\vect{c}_B/(\lVert\vect{c}_A\rVert\lVert\vect{c}_B\rVert)=@cosAB@$ entre A y B y
$@cosAC2@$ entre A y C. Dos colores grises, uno casi negro y el otro casi blanco, apuntan en la misma dirección de $\mathbb{R}^{3}$ (coseno $\approx1$)
aunque estén muy lejos entre sí (norma de la diferencia grande): el coseno mide la dirección, es decir el \emph{tono}, y la norma de la
diferencia mide cuánto cambia el color (figura~\ref{fig:vec-colores}). La segmentación usa esa diferencia para separar el casco del escritorio.

\begin{figure}[htbp]
\centering
\includegraphics[width=0.97\linewidth]{vec_colores.jpg}
\caption{Los colores de los píxeles A, B y C como vectores de $\mathbb{R}^{3}$ (izquierda), y norma de la diferencia y coseno del ángulo
entre cada par (derecha).}
\label{fig:vec-colores}
\end{figure}

\paragraph{Los vectores de un triángulo y el producto cruz.} Cada triángulo de la malla se describe con tres vértices, y dos de sus lados son
vectores, $\vect{e}_1=\vect{p}_1-\vect{p}_0$ y $\vect{e}_2=\vect{p}_2-\vect{p}_0$. Con los vértices del triángulo de ejemplo,
\begin{equation}
\vect{e}_1=@e1v@\ \text{mm},\qquad \vect{e}_2=@e2v@\ \text{mm}.
\end{equation}
El producto cruz $\vect{n}=\vect{e}_1\times\vect{e}_2$ da un vector perpendicular al triángulo; sus componentes son
@cruz@
es decir $\vect{n}=@nvec@$. Su norma es $\lVert\vect{n}\rVert=@nn@$, que es el doble del área del triángulo (área $=@area@$ mm$^{2}$), y
el vector unitario es $\hat{\vect{n}}=\vect{n}/\lVert\vect{n}\rVert=@nu@$. Como $\vect{n}$ es perpendicular al triángulo, su producto punto
con los dos lados debe ser cero: $\hat{\vect{n}}\cdot\vect{e}_1=@ne1@$ y $\hat{\vect{n}}\cdot\vect{e}_2=@ne2@$ (los ceros de la
máquina vienen de redondear a cuatro decimales). Y el producto punto de la normal con el rayo hacia el centro del triángulo,
$\vect{c}_t=\vect{p}_0+\vect{p}_1+\vect{p}_2=@cen@$, dice si la cámara lo ve de frente o de canto:
$\lvert\hat{\vect{n}}\cdot\hat{\vect{c}}_t\rvert=@cosv@$ (un valor cercano a $1$ es de frente y cercano a $0$, de canto).

\begin{figure}[htbp]
\centering
\includegraphics[width=0.97\linewidth]{vec_triangulo.jpg}
\caption{Triángulo de la malla del casco: los lados $\vect{e}_1$ y $\vect{e}_2$, la normal unitaria $\hat{\vect{n}}$ del producto cruz y
las operaciones con vectores (norma, producto punto con los lados y área).}
\label{fig:vec-triangulo}
\end{figure}

\paragraph{Del punto de la cámara al objeto.} Para pasar de los ejes de la cámara a los del objeto se resta el vector $\vect{c}$ (el centro
del sólido) y se gira con $R_o$. Con $\vect{c}=@c0@$ mm,
\begin{equation}
\vect{p}_{\text{obj}}=R_o(\vect{P}-\vect{c}),\qquad R_o=\operatorname{diag}(1,-1,-1),
\end{equation}
que cambia el signo de las componentes $y$ y $z$ (un giro de $180^\circ$ alrededor de $x$) sin cambiar la norma. En el cuadro~\ref{tab:objeto-vec}
se ve para los tres puntos de ejemplo.

\begin{table}[htbp]
\caption{Los tres puntos del casco en los ejes de la cámara $\vect{P}$, restados del centro $\vect{c}$ y girados $R_o(\vect{P}-\vect{c})$ (mm)}
\label{tab:objeto-vec}
\centering\footnotesize
\begin{tabular}{@{}cp{3.7cm}p{3.7cm}p{3.7cm}@{}}
\toprule
Punto & $\vect{P}$ & $\vect{P}-\vect{c}$ & $R_o(\vect{P}-\vect{c})$ \\
\midrule
@filas_obj@
\bottomrule
\end{tabular}
\end{table}
"""
    d = dict(
        uA=ej["visera"]["u"], vA=ej["visera"]["v"], cx=n(cx, 0), cy=n(cy, 0), f=n(f, 3),
        rA=vec(rA, 4), rAx=n(rA[0], 4), rAy=n(rA[1], 4), rCx=n(rC[0], 4), rCy=n(rC[1], 4),
        nA=n(nA, 4), nC=n(nC, 4), cosAz=n(1 / nA, 4), thA=n(np.degrees(np.arccos(1 / nA)), 2),
        filas_rayos="\n".join(filas), ZA=n(ej["visera"]["Z_frente"], 2), PA=vec(ej["visera"]["P"], 2),
        nPA=n(float(np.linalg.norm(ej["visera"]["P"])), 2),
        dotAC=n(dot, 4), cosAC=n(dot / nA / nC, 4), thAC=n(ang(rA, rC), 2),
        cAv=pmi([[int(x)] for x in cA]), cBv=pmi([[int(x)] for x in cB]), cCv=pmi([[int(x)] for x in cC]),
        dAB=pmi([[int(x)] for x in dAB]), d1=int(dAB[0]), d2=int(dAB[1]), d3=int(dAB[2]),
        nAB=n(float(np.linalg.norm(dAB)), 1), cosAB=n(cos_AB, 4), cosAC2=n(cos_AC, 4),
        e1v=vec(e1, 4), e2v=vec(e2, 4), cruz=cruz_bloque, nvec=vec(nvec, 4), nn=n(nn, 4), area=n(nn / 2, 4), nu=vec(nu, 4),
        ne1=sci(float(nu @ e1), 1), ne2=sci(float(nu @ e2), 1), cen=vec(cen, 3), cosv=n(cosv, 4),
        c0=vec(c0, 1), filas_obj="\n".join(filas_obj),
    )
    # los valores científicos con signo
    return fill(plantilla, d)


# =============================================================================
def habitacion(H, C):
    cam = H["camara"]
    K, Ki = np.array(cam["K"]), np.array(cam["K_inv"])
    f, cx, cy = cam["f"], cam["cx"], cam["cy"]
    W, Hh = H["foto"]["ancho"], H["foto"]["alto"]
    ej = H["ejemplos"]
    pl = H["planos"]
    pn = {p["nombre"]: p for p in pl}
    piso, techo = pn["piso"], pn["techo"]
    p1, p2, p3 = pn["pared 1"], pn["pared 2"], pn["pared 3"]
    rs = H["ransac_ejemplo"]
    rej = H["rejilla"]
    mal = H["malla"]
    rot = H["rotacion"]
    cl = H["clases"]
    prof = H["profundidad"]
    npiso, ntecho = np.array(piso["n"]), np.array(techo["n"])
    n1, n3 = np.array(p1["n"]), np.array(p3["n"])
    # tabla de clases
    est = cl["estructura"]
    orden = ["pared", "piso", "techo", "ventana", "espejo"]
    filas_cl = "\n".join(rf"{k.capitalize()} & ${N(est[k])}$ & ${n(cl['frac'][k] * 100, 1)}$\,\% \\" for k in orden if k in est)
    # píxeles de ejemplo
    nombres = [("piso", "Piso"), ("pared", "Pared"), ("techo", "Techo"), ("mueble", "Sofá")]
    filas_pix, cols_H = [], []
    for k, tt in nombres:
        e = ej[k]
        r = np.array(e["rayo"])
        cols_H.append((e["u"], e["v"]))
        filas_pix.append(rf"{tt} & $({e['u']},\,{e['v']})$ & $({n(r[0], 4)},\,{n(r[1], 4)},\,1)$ & ${n(e['norma_rayo'], 3)}$ & ${n(e['angulo_eje'], 1)}^\circ$ & ${n(e['Z_red'], 2)}$ \\")
    Hm = np.array([[u, v, 1] for u, v in cols_H], float).T            # 3 × 4
    KiH = Ki @ Hm
    Zs = np.array([ej[k]["Z_red"] for k, _ in nombres])
    Pm = KiH * Zs
    # RANSAC: e1, e2, cruz, n, d
    a = np.array(rs["puntos"])
    e1, e2 = np.array(rs["e1"]), np.array(rs["e2"])
    cruz_bloque, nvec = cruz_tex(e1, e2, 3)
    nn = float(np.linalg.norm(nvec))
    nu = nvec / nn
    nu = nu * (-1.0 if nu[1] > 0 else 1.0)       # apunta hacia arriba ($-y$)
    d0 = float(nu @ a[0])
    # planos: tabla
    filas_pl = []
    for p in pl:
        nn_ = np.array(p["n"])
        filas_pl.append(rf"{p['nombre'].capitalize()} & $({n(nn_[0], 4)},\,{n(nn_[1], 4)},\,{n(nn_[2], 4)})$ & ${n(p['d'], 3)}$ & ${n(p['angulo_vertical'], 1)}^\circ$ & ${N(p['n_pix'])}$ \\")
    # perpendicularidad
    dot_pp1 = float(npiso @ n1)
    dot_p13 = float(n1 @ n3)
    # intersección rayo–plano para tres píxeles
    filas_int = []
    for k, tt, plano in (("piso", "Piso", "piso"), ("pared", "Pared", "pared 1"), ("techo", "Techo", "techo")):
        e = ej[k]
        c = next(c for c in e["cortes"] if c["plano"] == plano)
        Pr, Pq = np.array(e["P_red"]), np.array(e["P_plano"])
        filas_int.append(rf"{tt} & {plano} & ${n(c['n_dot_r'], 4)}$ & ${n(pn[plano]['d'], 3)}$ & ${n(c['t'], 3)}$ & ${n(e['Z_red'], 3)}$ & ${n(float(np.linalg.norm(Pq - Pr)) * 100, 1)}$ \\")
    ep = ej["piso"]
    rr = np.array(ep["rayo"])
    A3 = np.array([[f, 0, cx - ep["u"]], [0, f, cy - ep["v"]], [npiso[0], npiso[1], npiso[2]]])
    b3 = np.array([0, 0, piso["d"]])
    Psol = np.linalg.solve(A3, b3)
    det_A3 = float(np.linalg.det(A3))
    nr = float(npiso @ rr)
    # cuarto vacío
    em = ej["mueble"]
    filas_vac = "\n".join(
        rf"{c['plano'].capitalize()} & ${n(c['n_dot_r'], 3)}$ & ${n(pn[c['plano']]['d'], 3)}$ & ${n(c['t'], 2)}$ & {'sí' if c['t'] > 0.1 else 'no'} \\"
        for c in em["cortes"])
    tmin = min(c["t"] for c in em["cortes"] if c["t"] > 0.1)
    # triángulos de ejemplo
    tj = H["triangulos_ejemplo"]
    tp_, ts_ = tj["plano"], tj["salto"]
    cr_p, nv_p = cruz_tex(tp_["e1"], tp_["e2"], 1)
    cr_s, nv_s = cruz_tex(ts_["e1"], ts_["e2"], 1)
    cos_p = abs(float(nv_p @ np.array(tp_["centro"]))) / float(np.linalg.norm(nv_p) * np.linalg.norm(tp_["centro"]))
    cos_s = abs(float(nv_s @ np.array(ts_["centro"]))) / float(np.linalg.norm(nv_s) * np.linalg.norm(ts_["centro"]))
    sen5 = float(np.sin(np.radians(rej["canto_grados"])))
    # rotación
    nR, mR, vR = np.array(rot["n"]), np.array(rot["m"]), np.array(rot["v"])
    plantilla = r"""
\subsection{Caso de estudio: una sala, paso a paso}\label{sec:sala}

Igual que con el casco, el proceso de las habitaciones se desarrolla aquí con una fotografía real y con los números que produce el programa. La
foto es una sala con un sofá, una mesa de centro y un mueble de televisión (figura~\ref{fig:hab-entrada}), tomada de Wikimedia Commons
(autor: Shixart1985, licencia CC BY 2.0, \texttt{commons.wikimedia.org}). Se reduce a $@W@\times@Hh@$ píxeles ($N_p=@Np@$ píxeles) y se procesa con el
programa tal como está (procesamiento bajo); cada etapa se ejecutó por separado
(\texttt{experimentos/habitacion\_paso\_a\_paso.py}). En la figura~\ref{fig:hab-flujo} están los seis pasos y los objetos matemáticos de cada uno.

\begin{figure}[htbp]
\centering
\includegraphics[width=0.95\linewidth]{hab_entrada.jpg}
\caption{Foto de la sala reducida a $@W@\times@Hh@$ píxeles, con los ejes $u$ (columna) y $v$ (fila), el centro $(c_x,c_y)$ y los cuatro píxeles
(piso, pared, techo y sofá) que se siguen en las cuentas.}
\label{fig:hab-entrada}
\end{figure}

\begin{figure}[htbp]
\centering
\includegraphics[width=\linewidth]{hab_flujo.jpg}
\caption{Los seis pasos de la reconstrucción de una habitación y el objeto matemático de cada uno.}
\label{fig:hab-flujo}
\end{figure}

\paragraph{La cámara de la sala.} Con el ángulo de visión $\alpha=65^\circ$ y $W=@W@$, la distancia focal es $f=(W/2)/\tan(\alpha/2)=@f@$ píxeles,
$(c_x,c_y)=(@cx@,@cy@)$ y
\begin{equation}
K=@K@,\qquad \det K=f^2=@detK@,
\end{equation}
\begin{equation}
K^{-1}=@Ki@ .
\end{equation}
Son los mismos valores que en el casco porque la foto tiene el mismo ancho; solo cambia $c_y$ porque la altura es otra.

\paragraph{Paso 1: profundidad en metros y clase de cada píxel.} La red Depth Anything V2 entrega una matriz $Z$ de $@Hh@\times@W@$ números con
la distancia en metros de cada píxel al plano de la cámara: va de @Zmin@ m (la mesa de centro) a @Zmax@ m (el fondo), con media de @Zmed@ m,
y tardó @tprof@ ms. SegFormer entrega otra matriz del mismo tamaño con la clase de cada píxel (@ncl@ clases distintas en esta foto, en @tseg@ ms); las
del cuarto se cuentan en el cuadro~\ref{tab:hab-clases} y las dos matrices se ven en la figura~\ref{fig:hab-etapas}.

\begin{table}[htbp]
\caption{Píxeles de la foto de la sala que pertenecen a cada superficie del cuarto}
\label{tab:hab-clases}
\centering\small
\begin{tabular}{@{}lrr@{}}
\toprule
Clase & Píxeles & Parte de la foto \\
\midrule
@filas_clases@
\bottomrule
\end{tabular}
\end{table}

\begin{figure}[htbp]
\centering
\includegraphics[width=0.97\linewidth]{hab_etapas.jpg}
\caption{De la foto a las matrices del proceso: foto, profundidad $Z(u,v)$ de la red, clase de cada píxel y planos ajustados (un color por plano).}
\label{fig:hab-etapas}
\end{figure}

\paragraph{De los píxeles a vectores.} Cada píxel se convierte en un vector con $\vect{r}=K^{-1}\vect{p}$ (como en la
sección «\nameref{sec:vectores-casco}») y luego en un punto $\vect{P}=Z\,\vect{r}$. El cuadro~\ref{tab:hab-rayos} muestra los cuatro píxeles de la figura~\ref{fig:hab-entrada},
y la retroproyección de los cuatro a la vez es un solo producto de matrices, $P=K^{-1}H\operatorname{diag}(Z_1,\dots,Z_4)$, con $H$ de $3\times4$ (cada
columna es un píxel):
\begin{equation}
H=@Hm@,
\end{equation}
\begin{equation}
K^{-1}H=@KiH@,
\end{equation}
\begin{equation}
P=K^{-1}H\begin{pmatrix}@z1@&0&0&0\\0&@z2@&0&0\\0&0&@z3@&0\\0&0&0&@z4@\end{pmatrix}=@Pm@\ \text{m} .
\end{equation}
Las columnas de $P$ son los puntos del piso, de la pared, del techo y del sofá en los ejes de la cámara.

\begin{table}[htbp]
\caption{Cuatro píxeles de la sala como vectores: píxel $(u,v)$, rayo $\vect{r}=K^{-1}\vect{p}$, su norma, el ángulo con el eje óptico y la profundidad $Z$ de la red (m)}
\label{tab:hab-rayos}
\centering\small
\begin{tabular}{@{}llcccc@{}}
\toprule
 & $(u,v)$ & $\vect{r}=K^{-1}\vect{p}$ & $\lVert\vect{r}\rVert$ & $\theta$ & $Z$ \\
\midrule
@filas_pix@
\bottomrule
\end{tabular}
\end{table}

\paragraph{Paso 2: ajustar planos a los puntos con RANSAC} El piso, el techo y las paredes son planos, es decir, conjuntos de puntos $\vect{P}$ que cumplen
$\vect{n}\cdot\vect{P}=d$ con $\vect{n}$ de norma $1$. Como la red tiene ruido, los puntos no caen exactamente en un plano, y se busca el
plano que más puntos explica con el método RANSAC, que repite tres pasos:
(1) tomar tres puntos al azar de la misma clase, (2) calcular el plano que pasa por ellos con el producto cruz y (3) contar cuántos de los demás puntos
están a menos de una tolerancia de ese plano. Para el piso, el primer intento con una muestra de $@nmues@$ puntos de la clase «piso» tomó los
píxeles @pix0@, @pix1@ y @pix2@, que en el espacio son
\begin{equation}
\vect{P}_0=@P0@,\quad \vect{P}_1=@P1@,\quad \vect{P}_2=@P2@\ \text{m},
\end{equation}
y dos lados del triángulo que forman son
\begin{equation}
\vect{e}_1=\vect{P}_1-\vect{P}_0=@e1@,\qquad \vect{e}_2=\vect{P}_2-\vect{P}_0=@e2@ .
\end{equation}
El producto cruz $\vect{n}=\vect{e}_1\times\vect{e}_2$ es perpendicular a los dos y, por tanto, al plano. Sus componentes son
@cruz@
es decir $\vect{n}=@nvec@$, con norma $\lVert\vect{n}\rVert=@nn@$. Se divide entre la norma para que quede de longitud $1$ y se elige el signo
para que apunte hacia arriba (el eje $y$ de la cámara apunta hacia abajo), y $d$ sale del producto punto con cualquiera de los tres puntos:
\begin{equation}
\hat{\vect{n}}=\frac{\vect{n}}{\lVert\vect{n}\rVert}=@nu@,\qquad d=\hat{\vect{n}}\cdot\vect{P}_0=@d0@\ \text{m}.
\end{equation}
La tolerancia es el 2\,\% de la mediana de $Z$ de la muestra, @tol@ cm, y @votos@ de los @de@ puntos de la muestra quedaron dentro de ella
(@vpct@\,\%): el plano es muy bueno. Con todos esos puntos se vuelve a ajustar el plano por mínimos cuadrados (el vector singular del menor valor
singular de los puntos centrados, el mismo SVD que se usa en la sección «\nameref{sec:inversa}») y queda el plano final del piso. El método repite esto
para el techo y hasta cuatro paredes. El cuadro~\ref{tab:hab-planos} reúne los seis planos y las figuras~\ref{fig:hab-ransac} y~\ref{fig:hab-planos} los muestran.

\begin{figure}[htbp]
\centering
\includegraphics[width=0.97\linewidth]{hab_ransac.jpg}
\caption{Izquierda: tres puntos del piso, los lados $\vect{e}_1$ y $\vect{e}_2$ de su triángulo y la normal $\hat{\vect{n}}$ del producto cruz.
Derecha: distancia de los puntos de la muestra a ese plano; los que quedan a la izquierda de la línea roja votan por él.}
\label{fig:hab-ransac}
\end{figure}

\begin{table}[htbp]
\caption{Planos ajustados en la sala: normal unitaria $\hat{\vect{n}}$, constante $d$ (m), ángulo de la normal con la vertical y píxeles que quedaron sobre el plano}
\label{tab:hab-planos}
\centering\small
\begin{tabular}{@{}lp{5.2cm}rrr@{}}
\toprule
Plano & $\hat{\vect{n}}=(n_x,n_y,n_z)$ & $d$ & Ángulo & Píxeles \\
\midrule
@filas_planos@
\bottomrule
\end{tabular}
\end{table}

\begin{figure}[htbp]
\centering
\includegraphics[width=\linewidth]{hab_planos.jpg}
\caption{Los puntos de cada plano, vistos desde arriba y de lado, con la cámara en el origen. El piso (azul) sube suavemente hacia el fondo, el techo
(naranja) y las paredes forman las otras caras del cuarto.}
\label{fig:hab-planos}
\end{figure}

\paragraph{Qué dicen las normales.} Las normales son vectores y el producto punto dice cómo se cruzan los planos. La del piso es
$\hat{\vect{n}}_{\text{piso}}=@npiso@$; la de la pared de la izquierda, $\hat{\vect{n}}_1=@n1@$. Su producto punto es
\begin{equation}
\hat{\vect{n}}_{\text{piso}}\cdot\hat{\vect{n}}_1=@dpp1@\quad\Longrightarrow\quad\theta=\arccos(@dpp1@)=@angpp1@^\circ,
\end{equation}
casi perpendiculares, como deben ser un piso y una pared. Entre la pared izquierda y la del fondo, $\hat{\vect{n}}_1\cdot\hat{\vect{n}}_3=@dp13@$ y el ángulo es
$@angp13@^\circ$: no son $90^\circ$ exactos porque el programa supone un ángulo de visión de $65^\circ$ y la red tiene error, y eso inclina un poco los
planos lejanos (figura~\ref{fig:hab-normales}). El techo es casi paralelo al piso ($@angpt@^\circ$). Además, el valor $|d|$ de la normal del piso es la
\emph{altura de la cámara}: $|d|=@dabs@$ m.

\begin{figure}[htbp]
\centering
\includegraphics[width=0.97\linewidth]{hab_normales.jpg}
\caption{Las normales de cuatro planos como vectores unitarios (izquierda) y el ángulo entre cada par de planos, calculado con el producto punto (derecha).}
\label{fig:hab-normales}
\end{figure}

\paragraph{Paso 3: intersección rayo--plano.} Para que los planos queden perfectamente planos, el píxel cambia su profundidad por la del plano. El
punto del plano sobre el rayo es $\vect{P}=t\,\vect{r}$ con $\hat{\vect{n}}\cdot\vect{P}=d$, y de ahí $t\,(\hat{\vect{n}}\cdot\vect{r})=d$, es decir
\begin{equation}
Z=\frac{d}{\hat{\vect{n}}\cdot\vect{r}} .
\end{equation}
Para el píxel del piso, $\vect{r}=@rr@$ y $\hat{\vect{n}}\cdot\vect{r}=(@nx@)(@rx@)+(@ny@)(@ry@)+(@nz@)(1)=@nr@$, así que
\begin{equation}
Z=\frac{@dpiso@}{@nr@}=@Zp@\ \text{m},\qquad \vect{P}=@Zp@\cdot@rr@=@Ppl@\ \text{m},
\end{equation}
frente a $Z=@Zr@$ m de la red (una diferencia de @difZ@\,\%). La figura~\ref{fig:hab-rayoplano} lo muestra en un corte lateral. El cuadro~\ref{tab:hab-int}
repite la cuenta para el piso, la pared y el techo. Las cuentas solo valen si $\hat{\vect{n}}\cdot\vect{r}\neq0$ (el rayo no es paralelo al plano) y $Z>0$ (el
plano está delante de la cámara).

\begin{figure}[htbp]
\centering
\includegraphics[width=0.95\linewidth]{hab_rayo_plano.jpg}
\caption{Corte lateral de la sala: el rayo $\vect{r}=K^{-1}\vect{p}$ del píxel del piso corta el plano $\hat{\vect{n}}\cdot\vect{P}=d$ en
$Z=d/(\hat{\vect{n}}\cdot\vect{r})$; la distancia $|d|$ del origen al plano es la altura de la cámara.}
\label{fig:hab-rayoplano}
\end{figure}

\begin{table}[htbp]
\caption{Intersección rayo--plano para tres píxeles: $\hat{\vect{n}}\cdot\vect{r}$, $d$ (m), $Z=d/(\hat{\vect{n}}\cdot\vect{r})$ (m), $Z$ de la red (m) y distancia entre el punto de la red y el del plano (cm)}
\label{tab:hab-int}
\centering\small
\begin{tabular}{@{}llrrrrr@{}}
\toprule
 & Plano & $\hat{\vect{n}}\cdot\vect{r}$ & $d$ & $Z$ del plano & $Z$ de la red & Dist. \\
\midrule
@filas_int@
\bottomrule
\end{tabular}
\end{table}

\paragraph{La misma cuenta como sistema de ecuaciones.} Hallar $\vect{P}=(X,Y,Z)$ en el rayo del píxel $(u,v)$ y en el plano $\hat{\vect{n}}\cdot\vect{P}=d$ es
un sistema lineal de $3\times3$: las dos filas de arriba son la proyección $K\vect{P}=Z\vect{p}$ reescrita, $fX+(c_x-u)Z=0$ y $fY+(c_y-v)Z=0$, y la de abajo es el plano.
Para el píxel del piso,
\begin{equation}
\underbrace{\begin{pmatrix}@f@&0&@cxu@\\0&@f@&@cyv@\\@nx@&@ny@&@nz@\end{pmatrix}}_{A}
\begin{pmatrix}X\\Y\\Z\end{pmatrix}=\begin{pmatrix}0\\0\\@dpiso@\end{pmatrix}.
\end{equation}
Su determinante es $\det A=f^{2}\,(\hat{\vect{n}}\cdot\vect{r})=(@f2@)(@nr@)=@detA@$ (se desarrolla por la primera columna y sale $f^2$ por el producto punto), que es distinto de
cero justamente cuando el rayo no es paralelo al plano; entonces $A$ es invertible y $\vect{P}=A^{-1}\vect{b}=@Psol@$ m, el mismo punto de arriba. Es decir: la intersección
rayo--plano es un sistema $3\times3$ con solución única, y la fórmula $Z=d/(\hat{\vect{n}}\cdot\vect{r})$ es su solución despejada.

\begin{figure}[htbp]
\centering
\includegraphics[width=0.97\linewidth]{hab_planitud.jpg}
\caption{Izquierda: desviación de los puntos de cada superficie respecto de su plano antes (puntos de la red) y después de la intersección rayo--plano (escala
logarítmica). Derecha: cuánto cambia la profundidad, en promedio, al reemplazarla por la del plano.}
\label{fig:hab-planitud}
\end{figure}

La figura~\ref{fig:hab-planitud} resume el efecto en los seis planos: antes, los puntos del piso se desviaban @dsp@ cm de su plano y los de las paredes entre @dsmin@ y @dsmax@ cm; después, la
desviación es menor que $10^{-3}$ cm (cero, hasta el redondeo). La profundidad de la red cambia entre @cmin@\,\% y @cmax@\,\% al pasar al plano.

\paragraph{El cuarto vacío.} Con los mismos planos se calcula qué hay \emph{detrás} de cada mueble. Para un píxel tapado, el rayo corta todos los planos del cuarto en
$t_k=d_k/(\hat{\vect{n}}_k\cdot\vect{r})$, y los valores negativos (planos detrás de la cámara) se descartan; el verdadero fondo es el menor $t_k$ positivo. Para el píxel del sofá,
$\vect{r}=@rsofa@$ y el cuadro~\ref{tab:hab-vacio} da los seis cortes; el menor positivo es @tmin@ m (el piso), muy lejos del sofá ($Z=@Zsofa@$ m),
de modo que detrás del sofá se pone el piso a esa profundidad (figura~\ref{fig:hab-vacio}).

\begin{table}[htbp]
\caption{Cortes del rayo del píxel del sofá con cada plano: $t_k=d_k/(\hat{\vect{n}}_k\cdot\vect{r})$ y si queda delante de la cámara}
\label{tab:hab-vacio}
\centering\small
\begin{tabular}{@{}lrrrr@{}}
\toprule
Plano & $\hat{\vect{n}}_k\cdot\vect{r}$ & $d_k$ (m) & $t_k$ (m) & $t_k>0$ \\
\midrule
@filas_vacio@
\bottomrule
\end{tabular}
\end{table}

\begin{figure}[htbp]
\centering
\includegraphics[width=0.97\linewidth]{hab_vacio.jpg}
\caption{Izquierda: cortes del rayo del píxel del sofá con los seis planos; se elige el menor positivo. Derecha: vista desde arriba del rayo, el sofá y el
fondo que queda detrás de él.}
\label{fig:hab-vacio}
\end{figure}

\paragraph{Paso 4: la malla y los triángulos que se cortan.} Sobre la foto se coloca una rejilla de un nodo cada @paso@ píxeles: $@nx_@\times@ny_@=@nodos@$ nodos que,
llevados al espacio con $\vect{P}=Z\,K^{-1}\vect{p}$, son las filas de la matriz de vértices $V$. Cada celda se divide en dos triángulos y en total hay
$2(@nx_@-1)(@ny_@-1)=@tsc@$ triángulos. Donde un mueble tapa una pared, $Z$ salta de golpe (figura~\ref{fig:hab-corte}, izquierda) y los triángulos que unen los dos lados
quedan estirados, como cortinas, y vistos casi de canto. Un triángulo se elimina si cumple las dos condiciones siguientes, calculadas con vectores:
\begin{itemize}
  \item se ve de canto: $\lvert\hat{\vect{n}}\cdot\hat{\vect{c}}_t\rvert<\operatorname{sen}5^\circ=@sen5@$, con $\hat{\vect{n}}$ su normal (producto cruz de dos lados) y $\hat{\vect{c}}_t$ el
        rayo hacia su centro, y
  \item la profundidad salta: $Z_{\max}/Z_{\min}>1{,}12$.
\end{itemize}
Con un triángulo del piso, de vértices en los píxeles @tp_pix@, los lados (en mm) son $\vect{e}_1=@tpe1@$ y $\vect{e}_2=@tpe2@$ y
@cruz_p@
es decir $\vect{n}=@nvp@$ de norma $@nnp@$; el vector hacia el centro es $\vect{c}_t=@tpc@$ ($\lVert\vect{c}_t\rVert=@tpnc@$), y
\begin{equation}
\lvert\hat{\vect{n}}\cdot\hat{\vect{c}}_t\rvert=\frac{@tpdot@}{(@nnp@)(@tpnc@)}=@tpcos@>@sen5@,\qquad \frac{Z_{\max}}{Z_{\min}}=@tpraz@<1{,}12,
\end{equation}
así que el triángulo \emph{se conserva}. Con un triángulo del salto entre el sofá y la pared, en los píxeles @ts_pix@,
$\vect{e}_1=@tse1@$ y $\vect{e}_2=@tse2@$ y
@cruz_s@
es decir $\vect{n}=@nvs@$ de norma $@nns@$; con $\vect{c}_t=@tsc_@$ ($\lVert\vect{c}_t\rVert=@tsnc@$),
\begin{equation}
\lvert\hat{\vect{n}}\cdot\hat{\vect{c}}_t\rvert=\frac{@tsdot@}{(@nns@)(@tsnc@)}=@tscos@<@sen5@,\qquad \frac{Z_{\max}}{Z_{\min}}=@tsraz@>1{,}12,
\end{equation}
y el triángulo \emph{se elimina}. De los @tsc@ triángulos se eliminaron @cort@ (@cortpct@\,\%) y quedaron @tfin@ en la capa de adelante.

\begin{figure}[htbp]
\centering
\includegraphics[width=0.97\linewidth]{hab_corte.jpg}
\caption{Izquierda: profundidad a lo largo de la fila $v=@vfila@$ de la foto; el salto del sofá a la pared del fondo es donde se cortan los triángulos. Derecha:
$|\hat{\vect{n}}\cdot\hat{\vect{c}}_t|$ de un triángulo del piso (se conserva) y de uno del salto (se elimina) frente al umbral $\operatorname{sen}5^\circ$.}
\label{fig:hab-corte}
\end{figure}

\paragraph{Paso 5: la capa de atrás.} Detrás de los muebles se agrega una segunda capa de triángulos con la profundidad del cuarto vacío, y la
red LaMa inventa su textura. En esta sala la capa de atrás aportó @natras@ vértices a los @nfrente@ de la capa de adelante, y la matriz final tiene $N=@Nv@$ vértices ($V\in\mathbb{R}^{@Nv@\times3}$) y
$M=@Mf@$ triángulos ($F\in\mathbb{N}^{@Mf@\times3}$).

\paragraph{Paso 6: enderezar la escena con una rotación.} El celular casi nunca está derecho, y la normal del piso medida no es exactamente vertical. La del piso es
$\hat{\vect{n}}=@nR@$ y la vertical es $\vect{m}=@mR@$ (hacia arriba, en los ejes de la cámara). Se busca la rotación $R$ que lleva $\hat{\vect{n}}$ a $\vect{m}$ con la
fórmula de Rodrigues, $R=I+[\vect{v}]_\times+[\vect{v}]_\times^{2}/(1+c)$, donde
\begin{equation}
\vect{v}=\hat{\vect{n}}\times\vect{m}=@vR@,\qquad c=\hat{\vect{n}}\cdot\vect{m}=@cR@,
\end{equation}
y el ángulo del giro es $\theta=\arccos c=@thR@^\circ$ (es decir, la cámara estaba inclinada $@thR@^\circ$). Las matrices son
\begin{equation}
[\vect{v}]_\times=@vx@,\qquad [\vect{v}]_\times^{2}=@vx2@ ,
\end{equation}
y, sumando con $1/(1+c)=@inv1c@$,
\begin{equation}
R=@Rm@ .
\end{equation}
La comprobación es que $R\hat{\vect{n}}=@Rn@=\vect{m}$, que $\det R=@detR@$ y que $R^{T}R=I$ (coincide con la identidad con diferencias menores que $10^{-8}$): la rotación no
deforma el cuarto. Después se compone con el cambio de ejes del visor $R_o=\operatorname{diag}(1,-1,-1)$ y todos los vértices se giran con un solo producto de matrices,
\begin{equation}
V_{\text{giro}}=V\,G^{T},\qquad G=R_oR=@G@ .
\end{equation}
Los puntos del piso tenían una dispersión de la coordenada vertical de $@sa@$ mm y quedan con $@sd@$ mm, alrededor de la altura $y=@yh@$ mm, que es la altura de la cámara sobre el
piso ($@alt@$ m, igual a $|d|$). Los vectores $\hat{\vect{n}}$, $\vect{m}$ y $\vect{v}$ y las cifras anteriores están en la figura~\ref{fig:vec-rodrigues}, y el antes y el después del piso, en la figura~\ref{fig:hab-rotacion}.

\begin{figure}[htbp]
\centering
\includegraphics[width=0.97\linewidth]{vec_rodrigues.jpg}
\caption{Los vectores de la rotación de Rodrigues: la normal medida del piso $\hat{\vect{n}}$, la vertical $\vect{m}$ y el eje de giro $\vect{v}=\hat{\vect{n}}\times\vect{m}$, perpendicular a los dos.}
\label{fig:vec-rodrigues}
\end{figure}

\begin{figure}[htbp]
\centering
\includegraphics[width=\linewidth]{hab_rotacion.jpg}
\caption{Puntos del piso antes de girar (inclinados, en los ejes de la cámara), después de girar con $G=R_oR$ (horizontales) y distribución de su altura.}
\label{fig:hab-rotacion}
\end{figure}

\paragraph{Los vectores de la habitación en un cuadro.} El cuadro~\ref{tab:vec-hab} reúne los vectores que aparecieron y la operación con la que se
obtuvo cada uno.

\begin{table}[htbp]
\caption{Vectores del proceso de una habitación y operación con la que se obtienen}
\label{tab:vec-hab}
\centering\small
\begin{tabular}{@{}p{2.5cm}p{4.0cm}p{3.1cm}p{4.2cm}@{}}
\toprule
Vector & Qué representa & Valor en la sala & Operación \\
\midrule
$\vect{r}=K^{-1}\vect{p}$ & dirección del rayo de un píxel & $@rrx@$ (píxel del piso) & producto matriz--vector; norma $@nrr@$ \\
$\vect{P}=Z\,\vect{r}$ & punto 3D del píxel & $@Pplano@$ m & producto por un escalar \\
$\vect{e}_1,\ \vect{e}_2$ & dos lados del triángulo de RANSAC & $\lVert\vect{e}_1\rVert=@ne1@$ m & resta de puntos \\
$\hat{\vect{n}}$ & normal de un plano & $@nu@$ & producto cruz y norma \\
$\hat{\vect{n}}_i\cdot\hat{\vect{n}}_j$ & ángulo entre dos planos & $@dpp1@$ (piso--pared) & producto punto \\
$t_k=d_k/(\hat{\vect{n}}_k\cdot\vect{r})$ & corte del rayo con un plano & $@tmin@$ m (detrás del sofá) & producto punto y división \\
$\vect{v}=\hat{\vect{n}}\times\vect{m}$ & eje del giro que endereza la sala & $@vRrow@$ & producto cruz \\
$\hat{\vect{n}}\cdot\hat{\vect{c}}_t$ & si un triángulo se ve de canto & $@tpcos@$ (piso), $@tscos@$ (salto) & producto punto con el rayo al centro \\
\bottomrule
\end{tabular}
\end{table}
"""
    pix = rs["pix"]
    dpiso = piso["d"]
    pm_ = lambda v_, d=4: vec(v_, d)  # noqa: E731
    d = dict(
        W=W, Hh=Hh, Np=N(H["foto"]["pixeles"]), f=n(f, 3), cx=n(cx, 0), cy=n(cy, 0),
        K=pm(K.tolist(), 3), Ki=pm(Ki.tolist(), 6), detK=nm(cam["det_K"], 1),
        Zmin=n(prof["Zmin"], 2), Zmax=n(prof["Zmax"], 2), Zmed=n(prof["Zmedia"], 2), tprof=N(prof["ms"]),
        ncl=cl["n_clases"], tseg=N(cl["ms"]), filas_clases=filas_cl, filas_pix=chr(10).join(filas_pix),
        Hm=pmi([[int(x) for x in fila] for fila in Hm.tolist()]), KiH=pm(KiH.tolist(), 4),
        z1=n(Zs[0], 2), z2=n(Zs[1], 2), z3=n(Zs[2], 2), z4=n(Zs[3], 2), Pm=pm(Pm.tolist(), 3),
        nmues=N(rs["de"]), pix0=rf"$({pix[0][0]},\,{pix[0][1]})$", pix1=rf"$({pix[1][0]},\,{pix[1][1]})$", pix2=rf"$({pix[2][0]},\,{pix[2][1]})$",
        P0=vec(a[0], 3), P1=vec(a[1], 3), P2=vec(a[2], 3), e1=vec(e1, 3), e2=vec(e2, 3), cruz=cruz_bloque,
        nvec=vec(nvec, 3), nn=n(nn, 3), nu=vec(nu, 4), d0=n(d0, 3), tol=n(rs["tolerancia"] * 100, 1),
        votos=N(rs["votos"]), de=N(rs["de"]), vpct=n(rs["votos"] / rs["de"] * 100, 1),
        filas_planos="\n".join(filas_pl),
        npiso=vec(npiso, 4), n1=vec(n1, 4), dpp1=n(dot_pp1, 4), angpp1=n(ang(npiso, n1), 1),
        dp13=n(dot_p13, 3), angp13=n(ang(n1, n3), 1), angpt=n(ang(npiso, ntecho), 1), dpiso=n(dpiso, 3), dabs=n(abs(dpiso), 3),
        rr=vec(rr, 4), nx=n(npiso[0], 4), ny=n(npiso[1], 4), nz=n(npiso[2], 4), rx=n(rr[0], 4), ry=n(rr[1], 4),
        nr=n(nr, 4), Zp=n(ep["Z_plano"], 3), Zr=n(ep["Z_red"], 3), Ppl=vec(ep["P_plano"], 3),
        difZ=n(abs(ep["Z_plano"] - ep["Z_red"]) / ep["Z_red"] * 100, 1),
        filas_int="\n".join(filas_int), cxu=n(cx - ep["u"], 0), cyv=n(cy - ep["v"], 0), f2=N(f ** 2),
        detA=nm(det_A3, 1), Psol=vec(Psol, 3), rsofa=vec(em["rayo"], 4), tmin=n(tmin, 2), Zsofa=n(em["Z_red"], 2),
        filas_vacio=filas_vac,
        dsp=n(piso["disp_antes_m"] * 100, 1), dsmin=n(min(p["disp_antes_m"] for p in pl[1:]) * 100, 1),
        dsmax=n(max(p["disp_antes_m"] for p in pl[1:]) * 100, 1),
        cmin=n(min(p["error_Z_antes_pct"] for p in pl), 1), cmax=n(max(p["error_Z_antes_pct"] for p in pl), 1),
        paso=rej["paso"], nx_=rej["nx"], ny_=rej["ny"], nodos=N(rej["nodos"]), tsc=N(rej["triangulos_sin_corte"]),
        sen5=n(sen5, 3), tp_pix=", ".join(f"$({u},{v})$" for u, v in tp_["nodos_uv"]), ts_pix=", ".join(f"$({u},{v})$" for u, v in ts_["nodos_uv"]),
        tpe1=vec(tp_["e1"], 1), tpe2=vec(tp_["e2"], 1), cruz_p=cr_p, nvp=vec(nv_p, 1), nnp=n(float(np.linalg.norm(nv_p)), 1),
        tpc=vec(tp_["centro"], 1), tpnc=n(tp_["norma_c"], 1), tpdot=N(abs(float(nv_p @ np.array(tp_["centro"])))), tpcos=n(cos_p, 3),
        tpraz=n(tp_["razon_Z"], 3),
        tse1=vec(ts_["e1"], 1), tse2=vec(ts_["e2"], 1), cruz_s=cr_s, nvs=vec(nv_s, 1), nns=n(float(np.linalg.norm(nv_s)), 1),
        tsc_=vec(ts_["centro"], 1), tsnc=n(ts_["norma_c"], 1), tsdot=N(abs(float(nv_s @ np.array(ts_["centro"])))), tscos=n(cos_s, 3),
        tsraz=n(ts_["razon_Z"], 2),
        cort=N(rej["cortados"]), cortpct=n(rej["cortados"] / rej["triangulos_sin_corte"] * 100, 2), tfin=N(rej["triangulos"]),
        vfila=ts_["nodos_uv"][0][1],
        natras=N(mal["atras"]), nfrente=N(mal["n_frente"]), Nv=N(mal["N"]), Mf=N(mal["M"]),
        nR=vec(nR, 5), mR=vec(mR, 1), vR=vec(vR, 5), cR=n(rot["c"], 6), thR=n(rot["angulo"], 2),
        vx=pm(rot["vx"], 5), vx2=pm(rot["vx2"], 5), inv1c=n(1 / (1 + rot["c"]), 4), Rm=pm(rot["R"], 5),
        Rn=vec(rot["R_n"], 4), detR=n(rot["det_R"], 6), 
        G=pm(rot["G"], 5), sa=n(rot["altura_y_antes_std_mm"], 1), sd=n(rot["altura_y_despues_std_mm"], 1),
        yh=('-' if rot["altura_y_despues_media_mm"] < 0 else '') + N(abs(rot["altura_y_despues_media_mm"])), alt=n(rot["altura_camara_m"], 2),
        rrx=vec(rr, 4), nrr=n(float(np.linalg.norm(rr)), 4), Pplano=vec(ep["P_plano"], 2), ne1=n(float(np.linalg.norm(e1)), 2),
        vRrow=vec(vR, 3), nu_=vec(nu, 3),
    )
    return fill(plantilla, d)


def resultados_hab(H):
    rot, mal, rej, pl = H["rotacion"], H["malla"], H["rejilla"], H["planos"]
    tam = mal["tam_mm"]
    plantilla = r"""
\subsubsection*{La sala de referencia}

Con la foto de la sala del desarrollo matemático (sección «\nameref{sec:sala}») el programa produjo la malla de la figura~\ref{fig:hab-nube}:
@Nv@ vértices, @Mf@ triángulos y un cuarto de $@tx@\times@ty@\times@tz@$ m en @tt@ s (en este computador, con la
red de profundidad y SegFormer en CPU). El cuadro~\ref{tab:hab-res} resume lo medido.

\begin{figure}[htbp]
\centering
\includegraphics[width=0.97\linewidth]{hab_nube.jpg}
\caption{Vértices de la malla de la sala con el color de la foto: visto desde el punto de la foto (izquierda) y girado (derecha), donde se ve que el piso, las paredes y el techo son
planos y que el sofá y la mesa quedan delante de la pared.}
\label{fig:hab-nube}
\end{figure}

\begin{table}[htbp]
\caption{Medidas de la reconstrucción de la sala de referencia}
\label{tab:hab-res}
\centering\small
\begin{tabular}{@{}p{7.2cm}r@{}}
\toprule
Medida & Valor \\
\midrule
Píxeles de la foto & @Np@ \\
Planos encontrados (piso, techo y @npar@ paredes) & @npl@ \\
Nodos de la rejilla de adelante & @nodos@ \\
Triángulos de la rejilla / eliminados por saltos & @tsc@ / @cort@ (@cortpct@\,\%) \\
Vértices de la capa de atrás & @natras@ \\
Vértices $N$ y triángulos $M$ de la malla & @Nv@ y @Mf@ \\
Altura de la cámara sobre el piso & @alt@ m \\
Inclinación de la cámara corregida con la rotación & @thR@$^\circ$ \\
Dispersión de la altura del piso antes y después de girar & @sa@ mm y @sd@ mm \\
Desviación del piso respecto de su plano antes y después & @dpa@ cm y menos de $10^{-3}$ cm \\
Tiempo total & @tt@ s \\
\bottomrule
\end{tabular}
\note{No hay una medida de referencia de la sala, así que estos valores muestran la coherencia interna (planitud, perpendicularidad), no el error contra el cuarto real.}
\end{table}
"""
    d = dict(Nv=N(mal["N"]), Mf=N(mal["M"]), tx=n(tam[0] / 1000, 1), ty=n(tam[1] / 1000, 1), tz=n(tam[2] / 1000, 1),
             tt=n(H["tiempo_total_s"], 1), Np=N(H["foto"]["pixeles"]), npar=len([p for p in pl if p["nombre"].startswith("pared")]), npl=len(pl),
             nodos=N(rej["nodos"]), tsc=N(rej["triangulos_sin_corte"]), cort=N(rej["cortados"]),
             cortpct=n(rej["cortados"] / rej["triangulos_sin_corte"] * 100, 2), natras=N(mal["atras"]), alt=n(rot["altura_camara_m"], 2),
             thR=n(rot["angulo"], 2), sa=n(rot["altura_y_antes_std_mm"], 1), sd=n(rot["altura_y_despues_std_mm"], 1),
             dpa=n(pl[0]["disp_antes_m"] * 100, 1))
    return fill(plantilla, d)


def main(ent_h, ent_c, salida):
    eh, ec, sal = Path(ent_h), Path(ent_c), Path(salida)
    H = json.loads((eh / "hab_valores.json").read_text())
    C = json.loads((ec / "valores.json").read_text())
    cols = np.load(eh.parent / "colores_vec.npy")
    (sal / "vectores_casco.tex").write_text(vectores_casco(C, cols), encoding="utf8")
    (sal / "habitacion_matematica.tex").write_text(habitacion(H, C), encoding="utf8")
    (sal / "habitacion_resultados.tex").write_text(resultados_hab(H), encoding="utf8")
    print("escrito en", sal)


if __name__ == "__main__":
    main(*sys.argv[1:4])
