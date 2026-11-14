"""Genera las secciones LaTeX del informe con los números reales del casco.

    servidor/.venv/bin/python experimentos/casco_latex.py SALIDA/ docs/secciones/ [alto.json]

Lee valores.json y arreglos.npz (casco_paso_a_paso.py) y, si existe, alto.json
(valores del procesamiento alto). Escribe construccion.tex, casco_matematica.tex,
casco_resultados.tex y marco_inversas.tex.
"""

import json
import sys
from fractions import Fraction
from pathlib import Path

import numpy as np


# ---------- formato ---------------------------------------------------------
def n(x, d=2):
    """Número con coma decimal para modo matemático."""
    s = f"{x:.{d}f}"
    return s.replace(".", "{,}")


def N(i):
    """Entero con espacio fino entre miles."""
    return f"{int(i):,}".replace(",", "\\,")


def nm(x, d=2):
    """Número con coma decimal y espacio fino entre miles (modo matemático)."""
    r = round(abs(x), d)
    ent, dec = f"{r:.{d}f}".split(".")
    ent = f"{int(ent):,}".replace(",", "\\,")
    return ("-" if x < 0 else "") + ent + ("{,}" + dec if d else "")


def sci(x, d=3):
    """Notación científica para modo matemático: 1{,}762\\times10^{-6}."""
    m, e = f"{x:.{d}e}".split("e")
    return m.replace(".", "{,}") + r"\times10^{" + str(int(e)) + "}"


def pm(filas, d=4, tipo="pmatrix"):
    def c(x):
        if isinstance(x, str):
            return x
        return str(int(round(x))) if abs(x - round(x)) < 1e-12 else n(x, d)
    cuerpo = r" \\ ".join(" & ".join(c(x) for x in f) for f in filas)
    return rf"\begin{{{tipo}}} {cuerpo} \end{{{tipo}}}"


def pmi(filas, tipo="pmatrix"):
    cuerpo = r" \\ ".join(" & ".join(str(x) for x in f) for f in filas)
    return rf"\begin{{{tipo}}} {cuerpo} \end{{{tipo}}}"


def vec(v, d=4):
    return pm([[x] for x in v], d)


def fill(texto, d):
    for k, v in d.items():
        texto = texto.replace(f"@{k}@", str(v))
    sobran = sorted(set(__import__("re").findall(r"@[A-Za-z0-9_]+@", texto)))
    assert not sobran, f"placeholders sin valor: {sobran}"
    return texto


# ---------- Gauss-Jordan con fracciones --------------------------------------
def gauss_jordan(A):
    """Devuelve la lista de pasos (descripción, matriz aumentada) del método de Gauss-Jordan."""
    m = len(A)
    M = [[Fraction(x) for x in fila] + [Fraction(int(i == j)) for j in range(m)] for i, fila in enumerate(A)]
    pasos = [("Matriz aumentada $[A\\mid I]$", [r[:] for r in M])]
    for c in range(m):
        p = next(r for r in range(c, m) if M[r][c] != 0)
        if p != c:
            M[c], M[p] = M[p], M[c]
            pasos.append((f"$F_{c+1}\\leftrightarrow F_{p+1}$", [r[:] for r in M]))
        piv = M[c][c]
        if piv != 1:
            M[c] = [x / piv for x in M[c]]
            pasos.append((f"$F_{c+1}\\leftarrow F_{c+1}/{fr(piv)}$", [r[:] for r in M]))
        for r in range(m):
            if r != c and M[r][c] != 0:
                k = M[r][c]
                M[r] = [a - k * b for a, b in zip(M[r], M[c])]
                signo = "-" if k > 0 else "+"
                coef = "" if abs(k) == 1 else fr(abs(k)) + "\\,"
                pasos.append((f"$F_{r+1}\\leftarrow F_{r+1}{signo}{coef}F_{c+1}$", [r_[:] for r_ in M]))
    return pasos


def fr(x: Fraction):
    return str(x.numerator) if x.denominator == 1 else rf"\tfrac{{{x.numerator}}}{{{x.denominator}}}"


def aumentada(M, m):
    def f(x):
        if x.denominator == 1:
            return str(x.numerator)
        s = "-" if x < 0 else ""
        return rf"{s}\tfrac{{{abs(x.numerator)}}}{{{x.denominator}}}"
    cols = "c" * m + "|" + "c" * m
    cuerpo = r" \\ ".join(" & ".join(f(x) for x in fila) for fila in M)
    return rf"\left(\begin{{array}}{{{cols}}} {cuerpo} \end{{array}}\right)"


# ---------- construcción ----------------------------------------------------
def construccion(v, extra):
    cam = v["camara"]
    f = cam["f"]
    f800 = 400 / np.tan(np.radians(32.5))
    f1600 = 800 / np.tan(np.radians(32.5))
    plantilla = r"""
\subsection{Construcción del sistema}\label{sec:construccion}

El sistema se construyó con tres piezas de bajo costo (cuadro~\ref{tab:piezas}):
una placa ESP32-CAM, un adaptador USB-serial y un computador portátil. La placa
lleva el sensor de imagen y un microcontrolador, pero no tiene puerto USB; por
eso se conecta al computador mediante el adaptador, que traduce el puerto serial
(UART) de la placa al USB del computador (figura~\ref{fig:hardware}).

\begin{figure}[htbp]
\centering
\includegraphics[width=\linewidth]{hardware_bloques.jpg}
\caption{Construcción del sistema: el sensor entrega los píxeles a la ESP32-CAM, que
los comprime a JPEG y los envía por el cable USB al servidor; la página web los
muestra y recibe el modelo 3D.}
\label{fig:hardware}
\end{figure}

\begin{table}[htbp]
\caption{Piezas del sistema y sus datos principales}
\label{tab:piezas}
\centering\small
\begin{tabular}{@{}p{3.6cm}p{4.4cm}p{6.2cm}@{}}
\toprule
Pieza & Qué hace & Dato principal \\
\midrule
Sensor OV2640 & Convierte la luz en una matriz de píxeles & $1\,600\times1\,200$ píxeles, píxel de $2{,}2\ \mu$m, área de $3{,}59\times2{,}68$ mm \citep{omnivision} \\
ESP32-CAM (AI Thinker) & Lee el sensor, comprime a JPEG y envía los bytes & Dos núcleos, memoria PSRAM externa, sin puerto USB propio \citep{aithinker} \\
Adaptador USB-serial (FTDI FT232R) & Pasa la señal UART al USB & Velocidad usada: $1\,000\,000$ baudios \\
Computador portátil & Ejecuta el servidor (Python) y la página web & Procesador de 8 núcleos, sin tarjeta gráfica dedicada \\
\bottomrule
\end{tabular}
\end{table}

\paragraph{Firmware.} La ESP32-CAM se programó con un firmware propio
(\texttt{firmware/camara\_esp32}). Al arrancar configura el sensor para entregar
JPEG, con un reloj de $20$ MHz, resolución SVGA ($800\times600$ píxeles) y
calidad de compresión 10, y guarda dos cuadros a la vez en la memoria PSRAM para
leer uno mientras el sensor captura el siguiente. Después repite sin parar: toma
el cuadro más reciente y escribe sus bytes, tal cual, en el puerto serial.
También acepta comandos de un carácter (por ejemplo, \texttt{h} y \texttt{v}
activan el espejo horizontal y el volteo vertical del sensor), que la página usa
para enderezar la imagen.

\paragraph{Protocolo y velocidad.} Cada cuadro es un archivo JPEG completo, que
siempre empieza con los bytes \texttt{FF D8} y termina con \texttt{FF D9}; el
servidor corta el flujo de bytes en esos marcadores y obtiene un cuadro por
cada par. El puerto serial usa 8 bits de datos, sin paridad y 1 bit de parada:
cada byte viaja con $10$ bits (los 8 de datos más el de inicio y el de parada).
Con $B$ baudios, la capacidad del cable es
\begin{equation}
\text{bytes por segundo} = \frac{B}{10} = \frac{1\,000\,000}{10} = 100\,000,
\qquad
\text{cuadros por segundo} \le \frac{100\,000}{S},
\end{equation}
donde $S$ es el tamaño del cuadro en bytes. Un cuadro de $47\,020$ bytes, como
el que se midió en las pruebas, permite como máximo $100\,000/47\,020\approx2{,}1$
cuadros por segundo; como el tamaño del JPEG cambia con la escena, el promedio
medido fue de $17$ cuadros en $6$ segundos, unos $2{,}8$ por segundo. A
$2\,000\,000$ baudios los cuadros llegaban con bytes dañados y no se podían decodificar,
por eso se dejó $1\,000\,000$.

\subsection{Adquisición de los datos: de la luz a la matriz $I$}\label{sec:adquisicion}

Esta sección sigue el camino de un dato, desde que la luz llega al sensor hasta
que el programa tiene la matriz de la foto. Todos los números son de la foto del
casco (sección «\nameref{sec:casco}»).

\paragraph{1. El sensor.} Cada píxel del sensor mide cuánta luz recibe a través
de un filtro rojo, verde o azul. Con la cámara ESP32-CAM se puede pedir $800\times600$
píxeles (SVGA) o hasta $1\,600\times1\,200$ (UXGA). La distancia focal en píxeles
depende de la resolución: con el ángulo de visión horizontal $\alpha=65^\circ$
que usa el programa, $f=(W/2)/\tan(\alpha/2)$ vale $@f800@$ píxeles a $800$ de ancho,
$@f960@$ a $960$ y $@f1600@$ a $1\,600$. En milímetros, esa focal corresponde a
$@f1600@\times2{,}2\ \mu\text{m}\approx@fmm@$ mm. Las hojas de datos de módulos con este
sensor traen focales entre $3{,}6$ y $4{,}2$ mm \citep{microchip,arducam}; con el ancho del sensor
($3{,}59$ mm), $2\arctan\bigl(1{,}795/f_{\text{mm}}\bigr)$ da ángulos de visión horizontales entre unos $46^\circ$ y $53^\circ$.
Por eso el valor de $65^\circ$ es una suposición del programa y no una medida de la cámara (se podría
afinar calibrándola), y es una de las razones por las que el tamaño en milímetros del modelo es aproximado
(limitaciones, sección de discusión): con un ángulo menor, la misma foto corresponde a una focal mayor y a puntos 3D más grandes.

\paragraph{2. De bytes a matriz: el JPEG.} Los bytes que llegan por el cable son
una imagen comprimida. Para volver a la matriz $I$ hay que deshacer tres
transformaciones, y las tres son productos de matrices cuya inversa se conoce:
\begin{enumerate}
  \item \textbf{Transformada del coseno (DCT).} La foto se corta en bloques de
        $8\times8$ píxeles. Para cada bloque $X$ (valores de brillo menos $128$) el
        compresor guarda la matriz de coeficientes
        \begin{equation}
        Y = C\,X\,C^{T},\qquad
        C_{kj}=\alpha_k\cos\frac{(2j+1)k\pi}{16},\quad
        \alpha_0=\sqrt{\tfrac18},\ \alpha_k=\tfrac12\ (k\ge1),
        \end{equation}
        donde $C$ es de $8\times8$. Esa matriz es \emph{ortogonal}: $CC^T=I$, así que
        su inversa es su transpuesta, y el decodificador recupera el bloque con
        $X=C^{T}\,Y\,C$. Con un bloque real del casco (la unión entre la visera oscura y la mentonera clara,
        columnas $@bu@$--$@bu7@$ y filas $@bv@$--$@bv7@$ de la foto):
        \begin{equation*}
        X=@Xb@ ,\qquad
        Y=@Yb@ .
        \end{equation*}
        El coeficiente $Y_{10}=@Y10@$ es el más grande: el bloque pasa de oscuro
        (arriba) a claro (abajo), que es la frecuencia vertical más baja. Aplicando
        $C^{T}YC$ y sumando $128$ se recupera $X$ con un error máximo de
        $@errdct@$ (solo redondeo del computador), y se comprueba $\max|CC^T-I|=@errC@$.
  \item \textbf{De YCbCr a RGB.} El JPEG guarda el brillo $Y'$ y dos diferencias de
        color $C_b,C_r$ (cada una en $[0,255]$, con $128$ como cero) mediante una matriz de $3\times3$:
        \begin{equation}
        \begin{pmatrix}Y'\\C_b-128\\C_r-128\end{pmatrix}=
        \underbrace{@Mycc@}_{T}\begin{pmatrix}R\\G\\B\end{pmatrix},\qquad
        T^{-1}=@Myccinv@ .
        \end{equation}
        La inversa $T^{-1}$ se calcula con el mismo método de Gauss--Jordan de la
        sección «\nameref{sec:inversa}». Para el píxel B de la foto, $(R,G,B)=@rgbB@$ da
        $(Y',C_b,C_r)=@ycB@$, y al multiplicar por $T^{-1}$ se recupera $(R,G,B)=@rgbBrec@$.
  \item \textbf{Cuantización.} Los coeficientes se redondean al comprimir; esta es la única
        pérdida de información, y la controla el parámetro de calidad.
\end{enumerate}
El resultado es la matriz $I\in\{0,\dots,255\}^{H\times W\times3}$: una matriz de
$H\times W$ entradas, cada una un vector $(r,g,b)\in\mathbb{R}^3$.

\begin{figure}[htbp]
\centering
\includegraphics[width=\linewidth]{casco_dct.jpg}
\caption{Bloque de $8\times8$ píxeles del casco (izquierda), su matriz de
coeficientes $Y=CXC^T$ (centro, en valor absoluto) y el bloque recuperado con
$C^{T}YC$ (derecha): la inversa de una matriz ortogonal es su transpuesta.}
\label{fig:dct}
\end{figure}

\paragraph{3. Reducción de tamaño.} Antes de calcular, la foto se reduce al ancho de
procesamiento $@w@$ píxeles. Con el factor $s=@w@/@w0@=@esc@$, cada píxel nuevo es el promedio
ponderado por área de los píxeles que cubre, que es una combinación lineal de
las entradas de $I$. La foto de $@w0@\times@h0@$ queda de $@w@\times@h@$ píxeles
(cada píxel nuevo cubre $1/s^2=@inv_s2@$ píxeles originales).

\paragraph{4. Los números de entrada.} La matriz reducida $I'$ tiene
$@h@\cdot@w@=@pix@$ píxeles y $@nums@$ números. Los tres píxeles que se siguen en
todo el desarrollo son (figura~\ref{fig:casco-entrada}):
\begin{center}\small
\begin{tabular}{cccc}
\toprule
Píxel & $(u,v)$ & Lugar en la foto & $(R,G,B)$ \\
\midrule
A & $(@uA@,\,@vA@)$ & centro de la visera (oscuro) & $@rgbA@$ \\
B & $(@uB@,\,@vB@)$ & parte superior del casco (blanco) & $@rgbB@$ \\
C & $(@uC@,\,@vC@)$ & borde izquierdo de la máscara & $@rgbC@$ \\
\bottomrule
\end{tabular}
\end{center}
La distancia entre dos colores es la norma de la diferencia de sus vectores:
$\lVert c_A-c_B\rVert=\lVert@dAB@\rVert=@normAB@$, que es grande porque A es casi negro y B casi
blanco, mientras que B y un píxel vecino dan una norma cercana a cero.
"""
    d = dict(
        f800=n(f800, 3), f960=n(f, 3), f1600=n(f1600, 1), fmm=n(f1600 * 2.2e-3, 2),
        w=extra["w"], w0=extra["w0"], h=extra["h"], h0=extra["h0"], esc=n(extra["w"] / extra["w0"], 1), inv_s2=n((extra["w0"] / extra["w"]) ** 2, 4),
        pix=N(extra["w"] * extra["h"]), nums=N(extra["w"] * extra["h"] * 3),
        bu=extra["bu"], bu7=extra["bu"] + 7, bv=extra["bv"], bv7=extra["bv"] + 7,
        Xb=pmi(extra["X"].tolist()), Yb=pmi(extra["Yr"].tolist()), Y10=int(extra["Yr"][1][0]),
        errdct=f"{extra['errdct']:.1e}".replace("e-", r"\times10^{-").replace(".", "{,}") + "}", errC=f"{extra['errC']:.1e}".replace("e-", r"\times10^{-").replace(".", "{,}") + "}",
        Mycc=pm(extra["T"], 6), Myccinv=pm(np.round(np.array(extra["Ti"]), 3).tolist(), 3),
        rgbA=pmi([[x] for x in extra["rgbA"]]).replace("pmatrix", "pmatrix"), rgbB=pmi([[x] for x in extra["rgbB"]]), rgbC=pmi([[x] for x in extra["rgbC"]]),
        ycB=pm([[x] for x in extra["ycB"]], 2), rgbBrec=pm([[x] for x in extra["rgbBrec"]], 2),
        uA=extra["pA"][0], vA=extra["pA"][1], uB=extra["pB"][0], vB=extra["pB"][1], uC=extra["pC"][0], vC=extra["pC"][1],
        dAB=pmi([[a - b] for a, b in zip(extra["rgbA"], extra["rgbB"])]), normAB=n(extra["normAB"], 1),
    )
    return fill(plantilla, d)


# ---------- inversas (marco teórico) ---------------------------------------
def marco_inversas():
    A = [[2, 0, 1], [1, 1, 0], [0, 1, 1]]
    pasos = gauss_jordan(A)
    Ai = np.linalg.inv(np.array(A, float))
    det = round(float(np.linalg.det(np.array(A, float))))
    cuerpo = []
    for desc, M in pasos:
        cuerpo.append(rf"\text{{{desc}}}\quad& {aumentada(M, 3)}")
    pasos_tex = (r"\begin{align*}" + "\n" + " \\\\[4pt]\n".join(cuerpo) + "\n" + r"\end{align*}")
    inv = [[Fraction(x).limit_denominator(100) for x in fila] for fila in Ai]
    invtex = r"\frac{1}{%d}" % det + pmi([[int(round(x * det)) for x in fila] for fila in Ai.tolist()])
    plantilla = r"""
\subsection{La matriz inversa y cómo se calcula}\label{sec:inversa}

Sea $A$ una matriz cuadrada de $m\times m$. Su \emph{inversa} $A^{-1}$ es la matriz que cumple
$AA^{-1}=A^{-1}A=I$, donde $I$ es la identidad. Si $A$ transforma un vector, $\vect{y}=A\vect{x}$,
la inversa deshace la transformación, $\vect{x}=A^{-1}\vect{y}$; por eso resolver un sistema $A\vect{x}=\vect{b}$
con $A$ invertible es calcular $\vect{x}=A^{-1}\vect{b}$. Una matriz tiene inversa si y solo si su
determinante es distinto de cero (sus filas son linealmente independientes). Tres propiedades
se usan en el proyecto:
\begin{itemize}
  \item $(AB)^{-1}=B^{-1}A^{-1}$: la inversa de un producto es el producto de las inversas en
        orden contrario (para deshacer ``primero $B$ y luego $A$'' hay que deshacer primero $A$).
  \item Una matriz triangular con diagonal distinta de cero es invertible y su inversa también es triangular.
  \item Una matriz ortogonal ($Q^{T}Q=I$) tiene $Q^{-1}=Q^{T}$.
\end{itemize}

\paragraph{Método de Gauss--Jordan.} Se escribe la matriz aumentada $[A\mid I]$ y se aplican operaciones
entre filas (intercambiar dos filas, multiplicar una fila por un número distinto de cero y sumarle a una fila
un múltiplo de otra) hasta convertir la parte izquierda en $I$; lo que queda a la derecha es $A^{-1}$,
porque cada operación equivale a multiplicar por una matriz elemental $E_i$ y al terminar
$E_k\cdots E_1A=I$, es decir $E_k\cdots E_1=A^{-1}$. Ejemplo con
$A=@A@$, cuyo determinante es $\det A=@det@$:
@pasos@
Por lo tanto
\begin{equation}
A^{-1}=@inv@ ,\qquad\text{y se comprueba que } AA^{-1}=I.
\end{equation}

\paragraph{Inversa de una matriz de rotación.} Una rotación $R$ de $3\times3$ tiene filas de norma 1 y
perpendiculares entre sí, así que $R^{T}R=I$ y $R^{-1}=R^{T}$: invertirla no cuesta ninguna
cuenta, solo intercambiar filas por columnas. Por ejemplo, la rotación de $90^\circ$ alrededor del eje $x$ es
\begin{equation}
R_x(90^\circ)=\begin{pmatrix}1&0&0\\0&0&-1\\0&1&0\end{pmatrix},\qquad
R_x(90^\circ)^{-1}=R_x(90^\circ)^{T}=\begin{pmatrix}1&0&0\\0&0&1\\0&-1&0\end{pmatrix}=R_x(-90^\circ).
\end{equation}

\paragraph{Coordenadas homogéneas y la inversa de una matriz de $4\times4$.} Una rotación $R$ seguida de una
traslación $\vect{t}$ y de un cambio de escala $s$ se escribe como una sola matriz de $4\times4$
que actúa sobre el punto $(x,y,z,1)$:
\begin{equation}
\mathcal{M}=\begin{pmatrix}sR&\vect{t}\\\vect{0}^{T}&1\end{pmatrix},\qquad
\mathcal{M}^{-1}=\begin{pmatrix}s^{-1}R^{T}&-s^{-1}R^{T}\vect{t}\\\vect{0}^{T}&1\end{pmatrix}.
\end{equation}
La inversa se obtiene resolviendo $\mathcal{M}\vect{p}'=\vect{p}$ por bloques: $\vect{p}=s^{-1}R^{T}(\vect{p}'-\vect{t})$.
Esta forma se usa en el cambio de ejes de los objetos y en el procesamiento alto, donde el modelo generativo entrega cada pieza con su
propia matriz de $4\times4$ (secciones «\nameref{sec:casco-ejes}» y «\nameref{sec:alto-matematica}»).
Ejemplo numérico: con $s=1$, $R=R_x(90^\circ)$ y $\vect{t}=(1,2,3)^{T}$,
\begin{equation}
\mathcal{M}=@M4@,\qquad
\mathcal{M}^{-1}=\begin{pmatrix}R^{T}&-R^{T}\vect{t}\\\vect{0}^{T}&1\end{pmatrix}=@M4i@,\qquad
\mathcal{M}\mathcal{M}^{-1}=@M4p@=I_4 ,
\end{equation}
pues $-R^{T}\vect{t}=@mRt@$. Con el punto $\vect{p}=(1,1,1)$: $\mathcal{M}(\vect{p},1)=@Mp@$, y $\mathcal{M}^{-1}$ de ese resultado devuelve el punto original, $(@pb@)$.

\paragraph{Cálculo con la matriz de la cámara.} La inversa que más se usa es la de la matriz de la cámara
$K$, que es triangular. Como $K$ tiene la forma $\left(\begin{smallmatrix}f&0&c_x\\0&f&c_y\\0&0&1\end{smallmatrix}\right)$,
basta una operación por fila (restar $c_x$ y $c_y$ veces la tercera fila, y dividir entre $f$) para
llegar a $K^{-1}$; el cálculo se hace con los valores del casco en la sección «\nameref{sec:casco-camara}».
"""
    Rx = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], float)
    t = np.array([1.0, 2.0, 3.0])
    M4 = np.eye(4)
    M4[:3, :3], M4[:3, 3] = Rx, t
    M4i = np.linalg.inv(M4)
    assert np.allclose(M4 @ M4i, np.eye(4))
    pp = np.array([1.0, 1.0, 1.0])
    pmov = M4 @ np.append(pp, 1)
    pback = M4i @ pmov
    assert np.allclose(pback[:3], pp)
    d = dict(A=pmi(A), det=det, pasos=pasos_tex, inv=invtex,
             M4=pm(M4.tolist(), 0), M4i=pm(M4i.tolist(), 0), M4p=pm((M4 @ M4i).tolist(), 0),
             mRt=pm([[x] for x in (-Rx.T @ t)], 0), Mp=pm([[x] for x in pmov], 0), pb=r",\," .join(str(int(round(x))) for x in pback[:3]))
    return fill(plantilla, d)


# ---------- desarrollo con el casco -------------------------------------------
def casco_matematica(v, extra, alto):
    cam, seg, pro, mal = v["camara"], v["segmentacion"], v["profundidad"], v["malla"]
    K, Ki = np.array(cam["K"]), np.array(cam["K_inv"])
    f, cx, cy = cam["f"], cam["cx"], cam["cy"]
    ej = {e["nombre"]: e for e in v["ejemplos"]}
    A_, B_, C_ = ej["visera"], ej["casco_arriba"], ej["cerca_del_borde"]
    W, H = v["foto"]["ancho"], v["foto"]["alto"]
    Ro = np.diag([1.0, -1.0, -1.0])
    c = np.array(mal["centro_camara"])
    PA = np.array(A_["P"])
    pA = Ro @ (PA - c)
    Mo = np.eye(4)
    Mo[:3, :3], Mo[:3, 3] = Ro, -Ro @ c
    Moi = np.linalg.inv(Mo)
    rA, rC = np.array(A_["rayo"]), np.array(C_["rayo"])
    cos_ac = float(rA @ rC / (np.linalg.norm(rA) * np.linalg.norm(rC)))
    ang_ac = float(np.degrees(np.arccos(cos_ac)))
    ang_A = float(np.degrees(np.arctan(np.hypot(rA[0], rA[1]))))
    # sistema 2×3 del píxel A
    uA, vA = A_["u"], A_["v"]
    Asis = [[f, 0, cx - uA], [0, f, cy - vA]]
    tri = mal["triangulo"]
    Vt = np.array(tri["V"])
    e1, e2 = np.array(tri["e1"]), np.array(tri["e2"])
    nn = np.array(tri["normal"])
    nu = np.array(tri["normal_unitaria"])
    caras = mal["caras"]
    E = 3 * mal["M"] // 2
    euler = mal["N"] - E + mal["M"]
    lote = v["lote"]
    Hm, Ki_H, Zs, Pm = np.array(lote["H"]), np.array(lote["K_inv_H"]), lote["Z"], np.array(lote["P"])
    # fórmula de la altura (sin el suavizado final) para el píxel A
    R = pro["R_mm"]
    hA_formula = 0.75 * R * (0.7 * A_["r"] + 0.3 * A_["p_s"])
    g = 0.10 * R
    # malla
    u_ancho = seg["u_max"] - seg["u_min"] + 1
    v_alto = seg["v_max"] - seg["v_min"] + 1
    paso = int(np.ceil(max(u_ancho, v_alto) / 300))
    tabla_prof = "\n".join(
        rf"{nom} & {n(e['d'], 2)} & {n(e['s'], 4)} & {n(e['p_s'], 4)} & {n(e['D'], 4)} & {n(e['r'], 4)} & {n(e['h'], 2)} & {n(e['Z_frente'], 2)} & {n(e['Z_atras'], 2)} \\"
        for nom, e in (("A", A_), ("B", B_), ("C", C_))
    )
    plantilla = r"""
\subsection{Caso de estudio: el casco de referencia}\label{sec:casco}

Para que la matemática no quede en el aire, todo el desarrollo que sigue se hace con una fotografía real y con
los números que produce el programa: el casco que el profesor pidió usar como objeto de prueba (figura~\ref{fig:casco-entrada}).
La foto se subió con el botón \textbf{Subir foto} de la página (procesamiento bajo) y los valores de esta sección
salen de ejecutar cada paso por separado (\texttt{experimentos/casco\_paso\_a\_paso.py}); el modelo que aparece en la
pestaña del escáner (figura~\ref{fig:visor-bajo}) coincide con ellos, con $N=@Nv@$ vértices y $M=@Mf@$ caras.

\begin{figure}[htbp]
\centering
\includegraphics[height=8.2cm]{casco_entrada.jpg}
\hfill
\includegraphics[height=8.2cm]{casco_recorte.jpg}
\caption{Foto de referencia reducida a $@w@\times@h@$ píxeles, con los ejes $u$ (columna) y $v$ (fila), el recuadro de
encuadre, el centro de la imagen $(c_x,c_y)$ y los tres píxeles A, B y C que se siguen en las cuentas (izquierda); objeto
recortado por la máscara (derecha).}
\label{fig:casco-entrada}
\end{figure}

\subsection{Matriz de la cámara y su inversa con los valores del casco}\label{sec:casco-camara}

El ángulo de visión horizontal que usa el programa es $\alpha=65^\circ$ y la foto mide $W=@w@$ píxeles de ancho
y $H=@h@$ de alto. La distancia focal en píxeles y el centro de la imagen son
\begin{align*}
f &= \frac{W/2}{\tan(\alpha/2)}=\frac{@w2@}{\tan(32{,}5^\circ)}=\frac{@w2@}{@tan@}=@f@\ \text{píxeles},\\
c_x&=\frac{W}{2}=@cx@,\qquad c_y=\frac{H}{2}=@cy@ .
\end{align*}
Con ellos la matriz de la cámara y su determinante son
\begin{equation}
K=@K@,\qquad \det K=f\cdot f\cdot1=f^2=@detK@\neq0,
\end{equation}
así que $K$ es invertible. Su inversa se calcula con Gauss--Jordan (sección «\nameref{sec:inversa}»), con solo tres
operaciones entre filas, porque $K$ ya es triangular:
\begin{align*}
\left(\begin{array}{ccc|ccc} @f@ & 0 & @cx@ & 1 & 0 & 0\\ 0 & @f@ & @cy@ & 0 & 1 & 0\\ 0&0&1&0&0&1\end{array}\right)
&\xrightarrow{\;F_1\leftarrow F_1-c_xF_3,\ F_2\leftarrow F_2-c_yF_3\;}
\left(\begin{array}{ccc|ccc} @f@ & 0 & 0 & 1 & 0 & -@cx@\\ 0 & @f@ & 0 & 0 & 1 & -@cy@\\ 0&0&1&0&0&1\end{array}\right)\\[4pt]
&\xrightarrow{\;F_1\leftarrow F_1/f,\ F_2\leftarrow F_2/f\;}
\left(\begin{array}{ccc|ccc} 1&0&0& @if@ & 0 & -@cxf@\\ 0&1&0& 0 & @if@ & -@cyf@\\ 0&0&1&0&0&1\end{array}\right).
\end{align*}
Entonces
\begin{equation}
K^{-1}=@Ki@ .
\end{equation}
La comprobación numérica $KK^{-1}=I$ da
\begin{equation}
KK^{-1}=@K@@Ki@=@KKi@ .
\end{equation}

\subsection{La máscara $M$ del casco}\label{sec:casco-mascara}

La red U$^2$-Net entrega para cada píxel un valor $\alpha_{vu}\in[0,255]$; la máscara es la matriz de $@h@\times@w@$ entradas
\begin{equation}
M_{vu}=\begin{cases}1&\text{si }\alpha_{vu}>127,\\0&\text{si no,}\end{cases}
\end{equation}
que luego se limpia (se quitan puntos sueltos, se rellenan huecos y se conserva la región conectada que más se
superpone con el recuadro). Para el casco, la suma de las entradas de $M$ es
$\sum M_{vu}=@area@$ píxeles, que es el @fraccion@\,\% de los $@pix@$ píxeles de la foto, y la región ocupa las columnas
$u\in[@umin@,@umax@]$ y las filas $v\in[@vmin@,@vmax@]$ (una caja de $@uan@\times@van@$ píxeles). Cada uno de esos píxeles
será un punto del espacio.

\subsection{Profundidad del casco: de la disparidad a $Z$}\label{sec:casco-profundidad}

Para cada píxel con $M_{vu}=1$ se calculan cuatro números (las ecuaciones del Paso~2 de la metodología). Con los
valores globales $d_{\max}=@dmax@$ píxeles (la mayor distancia al borde de la silueta),
$\text{mm/px}=Z_0/f=400/@f@=@mmpx@$ y $R=d_{\max}\cdot\text{mm/px}=@R@$ mm, y con
$D_{\text{borde}}=@Dborde@$ y $D_{\text{alto}}=@Dalto@$ (mediana de la disparidad en el borde y percentil 99 dentro del casco),
el píxel A se calcula así:
\begin{align*}
s&=\frac{d}{d_{\max}}=\frac{@dA@}{@dmax@}=@sA@, &
p(s)&=\sqrt{1-(1-s)^2}=\sqrt{1-(@unms@)^2}=@pA@,\\
r&=\frac{D-D_{\text{borde}}}{D_{\text{alto}}-D_{\text{borde}}}=\frac{@DA@-@Dborde@}{@Dalto@-@Dborde@}=@rA@, &
h&=0{,}75\,R\,(0{,}7\,r+0{,}3\,p)=0{,}75\cdot@R@\cdot@mezcla@=@hform@\ \text{mm}.
\end{align*}
El valor guardado, $h=@hA@$ mm, es el de la fórmula después de un suavizado gaussiano que quita escalones. Con
$g=0{,}10\,R=@g@$ mm (medio grosor):
\begin{align*}
Z_{\text{frente}}&=Z_0-g-h=400-@g@-@hA@=@ZfA@\ \text{mm},\\
Z_{\text{atrás}}&=Z_0+g+0{,}6\,h=400+@g@+0{,}6\cdot@hA@=@ZaA@\ \text{mm}.
\end{align*}
El cuadro~\ref{tab:casco-prof} reúne los tres píxeles, la figura~\ref{fig:casco-etapas} muestra las matrices completas
($I$, $M$, $D$ y $Z_{\text{frente}}$) y la figura~\ref{fig:casco-perfil} sigue la fila del píxel A.

\begin{table}[htbp]
\caption{Profundidad de los tres píxeles de ejemplo (longitudes en mm)}
\label{tab:casco-prof}
\centering\small
\begin{tabular}{@{}lrrrrrrrr@{}}
\toprule
Píxel & $d$ (px) & $s$ & $p(s)$ & $D$ & $r$ & $h$ & $Z_{\text{frente}}$ & $Z_{\text{atrás}}$ \\
\midrule
@tabla@
\bottomrule
\end{tabular}
\end{table}

\begin{figure}[htbp]
\centering
\includegraphics[width=\linewidth]{casco_etapas.jpg}
\caption{Matrices del casco: foto $I$, máscara $M$, disparidad $D(u,v)$ de la red y profundidad $Z_{\text{frente}}(u,v)$
en milímetros (solo dentro de la máscara).}
\label{fig:casco-etapas}
\end{figure}

\begin{figure}[htbp]
\centering
\includegraphics[width=0.82\linewidth]{casco_perfil.jpg}
\caption{Fila $v=@vA@$ (la del píxel A y el C): la disparidad $D$ (arriba) se convierte en las profundidades $Z_{\text{frente}}$ y
$Z_{\text{atrás}}$ (abajo); el espesor del sólido es la diferencia entre ambas.}
\label{fig:casco-perfil}
\end{figure}

\subsection{Retroproyección del casco: de píxeles a puntos 3D}\label{sec:casco-retro}

\paragraph{Un píxel.} El píxel A se escribe en coordenadas homogéneas, $\vect{p}_A=(u,v,1)^T=@pAh@$, y su rayo es
\begin{equation}
\vect{r}_A=K^{-1}\vect{p}_A=@Ki@@pAh@=@rAv@ .
\end{equation}
El punto 3D es el rayo multiplicado por la profundidad $Z_{\text{frente}}=@ZfA@$ mm:
\begin{equation}
\vect{P}_A=Z\,\vect{r}_A=@ZfA@\cdot@rAv@=@PA@\ \text{mm}.
\end{equation}
Es decir, A está a $@PAx@$ mm a la derecha, $@PAy@$ mm @abajoarriba@ del eje de la cámara y a $@PAz@$ mm al frente.
Para comprobarlo se proyecta de nuevo: $K\vect{P}_A=@KPA@=Z\,\vect{p}_A$, y al dividir entre $Z$ se recupera el píxel $(@uA@,@vA@,1)$.

\paragraph{Por qué hace falta $Z$: el sistema $2\times3$.} Las dos ecuaciones de la proyección, con $\vect{P}=(X,Y,Z)^T$ desconocido y
el píxel A conocido, forman el sistema homogéneo
\begin{equation}
\begin{pmatrix}f&0&c_x-u\\0&f&c_y-v\end{pmatrix}\begin{pmatrix}X\\Y\\Z\end{pmatrix}=
@Asis@\begin{pmatrix}X\\Y\\Z\end{pmatrix}=\begin{pmatrix}0\\0\end{pmatrix}.
\end{equation}
Su matriz tiene rango $2$ y hay $3$ incógnitas, así que queda una variable libre. Al despejar:
$X=\tfrac{u-c_x}{f}Z=@ax@\,Z$ y $Y=\tfrac{v-c_y}{f}Z=@ay@\,Z$, es decir, toda la solución es la recta
$\vect{P}=Z\,(@ax@,\,@ay@,\,1)^T=Z\,\vect{r}_A$: el rayo. El espacio nulo de la matriz es justamente el vector $\vect{r}_A$ que dio $K^{-1}\vect{p}_A$.

\paragraph{Vectores de los rayos.} La longitud del rayo A es $\lVert\vect{r}_A\rVert=@nrA@$ y su ángulo con el eje óptico es
$\arctan\sqrt{(@ax@)^2+(@ay@)^2}=@angA@^\circ$. El ángulo entre los rayos de A y C sale del producto punto:
\begin{equation}
\cos\theta=\frac{\vect{r}_A\cdot\vect{r}_C}{\lVert\vect{r}_A\rVert\,\lVert\vect{r}_C\rVert}=
\frac{@dot@}{@nrA@\cdot@nrC@}=@cosac@\ \Rightarrow\ \theta=@angac@^\circ .
\end{equation}

\paragraph{Muchos píxeles a la vez.} En el programa no se calcula píxel por píxel. Para cinco píxeles del casco, repartidos
a lo largo de la máscara, se apilan como columnas de una matriz $H$ de $3\times5$ y se multiplica una sola vez:
\begin{equation}
H=@Hm@,\qquad K^{-1}H=@KiH@ .
\end{equation}
Multiplicar a la derecha por la matriz diagonal $\operatorname{diag}(Z_1,\dots,Z_5)$, con
$(Z_1,\dots,Z_5)=@Zs@$ mm, escala cada columna por su profundidad:
\begin{equation}
P=K^{-1}H\,\operatorname{diag}(Z_1,\dots,Z_5)=@Pm@\ \text{mm}.
\end{equation}
Con los $@Nn@$ nodos de la malla el procedimiento es el mismo, con una matriz $H$ de $3\times@Nn@$.

\subsection{Cambio de ejes: del sistema de la cámara al del objeto}\label{sec:casco-ejes}

Los puntos $\vect{P}$ están en los ejes de la cámara ($X$ a la derecha, $Y$ hacia abajo, $Z$ hacia adelante). El visor 3D usa los ejes
del objeto ($x$ a la derecha, $y$ hacia arriba, $z$ hacia el observador). Se pasa de uno a otro con una traslación al centro del
objeto, $\vect{c}$ (el punto medio de la caja que contiene a todos los vértices), y una rotación de $180^\circ$ alrededor del eje $x$:
\begin{equation}
\vect{p}=R_o(\vect{P}-\vect{c}),\qquad R_o=@Ro@,\qquad \vect{c}=@cc@\ \text{mm}.
\end{equation}
$R_o$ es una rotación ($R_o^TR_o=I$ y $\det R_o=1$), su inversa es su transpuesta y, como es diagonal, $R_o^{-1}=R_o^T=R_o$. Para el píxel A:
\begin{equation}
\vect{p}_A=R_o\left(@PA@-@cc@\right)=R_o@dPc@=@pAfinal@\ \text{mm}.
\end{equation}

\paragraph{Con una sola matriz de $4\times4$.} La traslación y la rotación se pueden juntar en una matriz homogénea de $4\times4$ que actúa sobre
el punto con un $1$ agregado, $(\vect{P},1)$ (coordenadas homogéneas, igual que el píxel $(u,v,1)$):
\begin{equation}
\begin{pmatrix}\vect{p}\\1\end{pmatrix}=\mathcal{M}_o\begin{pmatrix}\vect{P}\\1\end{pmatrix},\qquad
\mathcal{M}_o=\begin{pmatrix}R_o&-R_o\vect{c}\\\vect{0}^{T}&1\end{pmatrix}=@Mo@ .
\end{equation}
Su determinante es $\det\mathcal{M}_o=\det R_o=1$ (una matriz de rotación y traslación no cambia volúmenes), y su inversa se obtiene por bloques,
porque $-R_o^{T}(-R_o\vect{c})=\vect{c}$:
\begin{equation}
\mathcal{M}_o^{-1}=\begin{pmatrix}R_o^{T}&\vect{c}\\\vect{0}^{T}&1\end{pmatrix}=@Moi@,\qquad
\mathcal{M}_o\,\mathcal{M}_o^{-1}=@MMi@=I_4 .
\end{equation}
Con el píxel A, $\mathcal{M}_o(\vect{P}_A,1)=@pA4@=(\vect{p}_A,1)$, y al revés, $\mathcal{M}_o^{-1}(\vect{p}_A,1)=@PA4@=(\vect{P}_A,1)$: la inversa
devuelve el punto a los ejes de la cámara.

La figura~\ref{fig:casco-corte} muestra un corte lateral de los puntos del frente y de la parte de atrás y los rayos que los generan.

\begin{figure}[htbp]
\centering
\includegraphics[width=0.78\linewidth]{casco_corte.jpg}
\caption{Corte lateral (plano $Y$--$Z$) a lo largo de la columna $u=@uB@$: cada punto del frente está sobre el rayo de su
píxel, a la profundidad $Z_{\text{frente}}$; la parte de atrás queda a $Z_{\text{atrás}}$.}
\label{fig:casco-corte}
\end{figure}

\subsection{Construcción de la malla $V$, $F$ del casco}\label{sec:casco-malla}

Sobre la caja de $@uan@\times@van@$ píxeles se coloca una rejilla con un nodo cada $@paso@$ píxeles (el paso es $\lceil\max(@uan@,@van@)/300\rceil=@paso@$, para
tener a lo sumo 300 nodos por lado). Los nodos que caen dentro de la máscara son $@Nn@$ ($\approx@area@/@paso2@$, pues cada nodo representa
$@paso@\times@paso@$ píxeles). Cada uno se retroproyecta dos veces: una a $Z_{\text{frente}}$ y otra a $Z_{\text{atrás}}$, así que
$V$ tiene $2\cdot@Nn@=@Nv@$ filas. Las primeras filas de $V$ (en mm, ejes del objeto) y de $F$ (índices de fila de $V$) son
\begin{equation}
V=@Vej@\ \text{mm},\qquad F=@Fej@ .
\end{equation}
Cada celda de la rejilla con sus cuatro esquinas dentro del casco se divide en dos triángulos; la parte de atrás repite los mismos
triángulos con el orden invertido, y las paredes unen el contorno. Quedan @cfrente@ caras en el frente, @catras@ atrás y @cpared@ en las
paredes: $M=@Mf@$ triángulos en total.

\paragraph{Comprobación de que la malla es un sólido cerrado.} En una malla cerrada de triángulos cada arista es de dos caras, así que el número de aristas es
$E=3M/2=@E@$. La característica de Euler de una superficie cerrada sin agujeros (como una esfera) es $\chi=V-E+F=2$; aquí
$\chi=@Nv@-@E@+@Mf@=@euler@$.

\paragraph{Un triángulo en detalle.} Se toma el triángulo $@trii@$ del frente (el más cercano al centro del casco), de la fila $F_{@trii@}=@Ftri@$ de $F$
(figura~\ref{fig:casco-malla-zoom}). Sus vértices, como filas de una matriz de $3\times3$, y los dos lados que salen de $\vect{p}_0$ son
\begin{equation}
\begin{pmatrix}\vect{p}_0\\\vect{p}_1\\\vect{p}_2\end{pmatrix}=@Vt@,\qquad
\vect{e}_1=\vect{p}_1-\vect{p}_0=@e1@,\qquad \vect{e}_2=\vect{p}_2-\vect{p}_0=@e2@ .
\end{equation}
Su normal es el producto cruz, que se calcula con el determinante simbólico de la regla de Sarrus:
\begin{equation}
\vect{n}=\vect{e}_1\times\vect{e}_2=@ncruz@=@nn@,\qquad
\hat{\vect{n}}=\frac{\vect{n}}{\lVert\vect{n}\rVert}=\frac{1}{@nnorm@}@nn@=@nu@ .
\end{equation}
Como $\hat n_z>0$, la cara mira hacia el observador (hacia $+z$), que es lo correcto para el frente. El volumen del tetraedro que forma este
triángulo con el origen es la sexta parte del determinante de la matriz de sus vértices:
\begin{equation}
\frac{1}{6}\det\begin{pmatrix}\vect{p}_0\\\vect{p}_1\\\vect{p}_2\end{pmatrix}=\frac{@det@}{6}=@vol@\ \text{mm}^3 .
\end{equation}
Sumando esa cantidad sobre las $M$ caras se obtiene el volumen del casco,
$V_{\text{casco}}=\frac16\sum\det[\vect{p}_0\ \vect{p}_1\ \vect{p}_2]=@vtot@\ \text{mm}^3\approx@vlitros@$ litros.

\begin{figure}[htbp]
\centering
\includegraphics[width=0.62\linewidth]{casco_malla_zoom.jpg}
\caption{Zona de la malla $F$ alrededor del triángulo del ejemplo (en naranja), con sus vértices $\vect{p}_0$, $\vect{p}_1$ y $\vect{p}_2$.}
\label{fig:casco-malla-zoom}
\end{figure}

\paragraph{Resumen: matrices, vectores e inversas del proceso.} El cuadro~\ref{tab:resumen-algebra} reúne los objetos
matemáticos que aparecieron, su tamaño con el casco y cómo se usa o se invierte cada uno.

\begin{table}[htbp]
\caption{Matrices, vectores e inversas del proceso, con los valores del casco}
\label{tab:resumen-algebra}
\centering\small
\begin{tabular}{@{}p{2.5cm}p{3.1cm}p{3.7cm}p{4.8cm}@{}}
\toprule
Objeto & Tamaño & Valor en el casco & Inversa o uso \\
\midrule
$I$ (foto) & $@h@\times@w@\times3$ & @nums@ números & el JPEG se decodifica con $C^{T}$ y $T^{-1}$ \\
$M$ (máscara) & $@h@\times@w@$ & @area@ píxeles con 1 & filtra los píxeles que se retroproyectan \\
$K$ & $3\times3$ & $f=@f@$, $(c_x,c_y)=(@cx@,@cy@)$ & $K^{-1}$ con $1/f=@if@$, $-c_x/f=-@cxf@$, $-c_y/f=-@cyf@$ \\
$\vect{r}$ (rayo) & $3\times1$ & $\vect{r}_A=@rAcorto@$ & $K^{-1}\vect{p}$; norma $@nrA@$ \\
$H$ & $3\times N$ & $3\times@Nn@$ & $K^{-1}H\operatorname{diag}(Z)$ en un solo producto \\
$\operatorname{diag}(Z)$ & $N\times N$ & $Z\in[@zmin@,@zmax@]$ mm & inversa $\operatorname{diag}(1/Z)$ \\
$R_o$ & $3\times3$ & $\operatorname{diag}(1,-1,-1)$ & $R_o^{-1}=R_o^{T}=R_o$ \\
$V$, $F$ & $N\times3$, $M\times3$ & $@Nv@$ y $@Mf@$ filas & $\tfrac16\sum\det$ da el volumen: @vlitros@ L \\
$\mathcal{M}$ (alto) & $4\times4$ & rotación $R_x(90^\circ)$ & $\mathcal{M}^{-1}=\mathcal{M}^{T}$ \\
\bottomrule
\end{tabular}
\end{table}
"""
    d = dict(
        Nv=N(mal["N"]), Mf=N(mal["M"]), w=W, h=H, w2=W // 2, tan=n(cam["tan_medio_fov"], 5), f=n(f, 3), cx=n(cx, 0), cy=n(cy, 0),
        K=pm(K.tolist(), 3), Ki=pm(Ki.tolist(), 6), KKi=pm(np.array(cam["K_por_Kinv"]).tolist(), 1).replace("{,}0", "{,}0"),
        detK=N(int(cam["det_K"])) + "{,}" + f"{cam['det_K'] % 1:.1f}"[2:], **{"if": n(1 / f, 6), "cxf": n(cx / f, 6), "cyf": n(cy / f, 6)},
        area=N(seg["area_px"]), fraccion=n(seg["fraccion"] * 100, 2), pix=N(W * H), umin=seg["u_min"], umax=seg["u_max"], vmin=seg["v_min"], vmax=seg["v_max"],
        uan=u_ancho, van=v_alto,
        dmax=n(pro["d_max_px"], 3), mmpx=n(pro["mm_por_px"], 4), R=n(R, 3), Dborde=n(pro["D_borde"], 4), Dalto=n(pro["D_alto"], 4),
        dA=n(A_["d"], 3), sA=n(A_["s"], 5), unms=n(1 - A_["s"], 5), pA=n(A_["p_s"], 5), DA=n(A_["D"], 4), rA=n(A_["r"], 5) if False else n(A_["r"], 5),
        mezcla=n(0.7 * A_["r"] + 0.3 * A_["p_s"], 5), hform=n(hA_formula, 2), hA=n(A_["h"], 2), g=n(g, 3), ZfA=n(A_["Z_frente"], 2), ZaA=n(A_["Z_atras"], 2),
        tabla=tabla_prof, rAv=vec(A_["rayo"], 6), vA=vA, uA=uA, uB=B_["u"],
        pAh=pm([[x] for x in A_["pixel_h"]], 0), PA=vec(A_["P"], 3), PAx=n(abs(A_["P"][0]), 2), PAy=n(abs(A_["P"][1]), 2), PAz=n(A_["P"][2], 2),
        abajoarriba="arriba" if A_["P"][1] < 0 else "abajo",
        KPA=vec(A_["K_P"], 1), Asis=pm(Asis, 3), ax=n(A_["rayo"][0], 6), ay=n(A_["rayo"][1], 6),
        nrA=n(np.linalg.norm(rA), 5), nrC=n(np.linalg.norm(rC), 5), angA=n(ang_A, 3), dot=n(float(rA @ rC), 5), cosac=n(cos_ac, 5), angac=n(ang_ac, 2),
        Hm=pm(Hm.tolist(), 0), KiH=pm(Ki_H.tolist(), 4), Zs=pm([Zs], 2), Pm=pm(Pm.tolist(), 2),
        Ro=pm(Ro.tolist(), 0).replace("{,}0", ""), cc=vec(c, 3), dPc=vec((PA - c), 3), pAfinal=vec(pA, 3),
        Mo=pm(Mo.tolist(), 3), Moi=pm(Moi.tolist(), 3), MMi=pm((Mo @ Moi).tolist(), 1),
        pA4=vec(list(Mo @ np.append(PA, 1)), 3), PA4=vec(list(Moi @ np.append(pA, 1)), 3),
        paso=paso, paso2=paso * paso, Nn=N(mal["n_frente"]),
        Vej=pm(mal["ejemplo_V"], 3), Fej=pmi(mal["ejemplo_F"]),
        cfrente=N(caras["frente"]), catras=N(caras["atras"]), cpared=N(caras["paredes"]), E=N(E), euler=euler,
        trii=N(tri["indice"]), Ftri=pmi([tri["F"]], "pmatrix").replace(r"\begin{pmatrix}", r"(").replace(r"\end{pmatrix}", ")").replace(" & ", ",\\,"),
        Vt=pm(Vt.tolist(), 4), e1=vec(e1, 4), e2=vec(e2, 4),
        ncruz=(rf"\begin{{vmatrix}}\vect{{i}}&\vect{{j}}&\vect{{k}}\\ {n(e1[0],4)}&{n(e1[1],4)}&{n(e1[2],4)}\\ {n(e2[0],4)}&{n(e2[1],4)}&{n(e2[2],4)}\end{{vmatrix}}"),
        nn=vec(nn, 4), nnorm=n(float(np.linalg.norm(nn)), 4), nu=vec(nu, 4),
        det=n(tri["det"], 3), vol=n(tri["volumen_tetraedro"], 3), vtot=N(round(mal["volumen_mm3"])), vlitros=n(mal["volumen_mm3"] / 1e6, 2),
        rAcorto="(" + r",\," .join(n(x, 3) for x in A_["rayo"]) + ")^{T}", zmin=n(pro["Zfrente_min"], 0), zmax=n(pro["Zfrente_max"], 0),
        nums=N(W * H * 3),
    )
    return fill(plantilla, d)


def alto_matematica(a, v):
    if not a:
        return ""
    q = a["q"]
    plantilla = r"""
\subsection{Procesamiento alto con el mismo casco}\label{sec:alto-matematica}

Con el selector de procesamiento en \textbf{Alto}, la misma foto se manda a un modelo generativo de imagen a 3D (Hunyuan3D, en la nube de fal.ai).
Antes de enviarla, el programa recorta el casco con la máscara $M$ de la sección «\nameref{sec:casco-mascara}», lo centra en un cuadrado blanco
con un margen del 15\,\% (así se parece a las fotos de producto con que se entrenan estos modelos) y lo escala a lo sumo a $1\,536\times1\,536$ píxeles.
El modelo devuelve un archivo glTF binario (GLB) con la malla y su textura; el programa lo recibió en @tiempo@ s y lo lee con los mismos objetos del
informe: una matriz de vértices $V$ y una de caras $F$ (cuadro~\ref{tab:alto}).

\begin{table}[htbp]
\caption{Casco con procesamiento bajo y con procesamiento alto}
\label{tab:alto}
\centering\small
\begin{tabular}{@{}lrr@{}}
\toprule
 & Bajo & Alto \\
\midrule
Vértices $N$ & @Nb@ & @Na@ \\
Triángulos $M$ & @Mb@ & @Ma@ \\
Tamaño (ancho $\times$ alto $\times$ fondo) & $@tb@$ mm & $@ta@$ u.m. \\
Parte de atrás & supuesta (extrusión) & generada por el modelo \\
Tiempo & @tpb@ s & @tpa@ s \\
\bottomrule
\end{tabular}
\par\smallskip\footnotesize u.m.: unidades del modelo (el modelo generativo no entrega escala real, por eso la ficha de la página no muestra medidas).
\end{table}

\paragraph{Álgebra lineal del resultado.} El archivo GLB trae cada malla con su propia transformación de nodo. En este caso no es una matriz
sino una rotación escrita como cuaternión unitario $\vect{q}=(x,y,z,w)=(@q0@,\,0,\,0,\,@q3@)$, que el programa convierte en matriz con la fórmula
\begin{equation}
R=\begin{pmatrix}1-2(y^2+z^2)&2(xy-zw)&2(xz+yw)\\2(xy+zw)&1-2(x^2+z^2)&2(yz-xw)\\2(xz-yw)&2(yz+xw)&1-2(x^2+y^2)\end{pmatrix}
=\begin{pmatrix}1&0&0\\0&0&-1\\0&1&0\end{pmatrix}=R_x(90^\circ),
\end{equation}
pues $x=w=@q0@=\sqrt{2}/2$ da $1-2x^2=0$ y $2xw=1$. Es la rotación de $90^\circ$ alrededor del eje $x$ del marco teórico,
que Hunyuan3D usa para que el eje $y$ quede hacia arriba. Escrita como matriz homogénea de $4\times4$ (con $s=1$ y $\vect{t}=\vect{0}$):
\begin{equation}
\mathcal{M}=@Mnodo@,\qquad \mathcal{M}^{-1}=\mathcal{M}^{T}=@Minv@ .
\end{equation}
Para saber dónde cae cada vértice en la escena, el programa multiplica las $8$ esquinas de la caja que contiene a cada malla (los valores mínimo y máximo de sus
coordenadas) por $\mathcal{M}$; así la caja del modelo queda de $@ta@$ unidades del modelo. La inversa $\mathcal{M}^{-1}=\mathcal{M}^{T}$ (aquí es solo
una rotación, sin traslación ni escala) devuelve las coordenadas al sistema original del modelo, que es el de la textura.
"""
    d = dict(tiempo=n(a["tiempo_s"], 1), Nb=N(v["malla"]["N"]), Mb=N(v["malla"]["M"]), Na=N(a["N"]), Ma=N(a["M"]),
             tb=r" \times ".join(n(x, 0) for x in v["malla"]["tam_mm"]), ta=r" \times ".join(n(x, 2) for x in a["tam_u"]),
             tpb=n(v["tiempos_ms"]["total"] / 1000, 1), tpa=n(a["tiempo_s"], 1),
             q0=n(q[0], 4), q3=n(q[3], 4),
             Mnodo=pm(a["M_nodo"], 0), Minv=pm(np.array(a["M_nodo"]).T.tolist(), 0))
    return fill(plantilla, d)


# ---------- resultados --------------------------------------------------------
def casco_resultados(v, alto):
    tm = v["tiempos_ms"]
    alg = tm["malla"]
    total = tm["total"]
    plantilla = r"""
\subsection{Reconstrucción del casco de referencia}\label{sec:res-casco}

La foto del casco que se usó en la sección «\nameref{sec:casco}» se subió a la página con el botón \textbf{Subir foto} (figura~\ref{fig:visor-bajo}).
Con procesamiento bajo el resultado se obtuvo en @total@ s con el computador de pruebas; el modelo tiene $N=@Nv@$ vértices, $M=@Mf@$ triángulos,
mide @tam@ mm y encierra un volumen de @vol@ litros. La figura~\ref{fig:casco-tiempos} muestra cuánto tarda cada etapa: la segmentación y la profundidad
(donde pesa la red neuronal) ocupan casi todo el tiempo, y los pasos de álgebra lineal ($K^{-1}$, la matriz $3\times N$ y la malla) suman @alg@ ms.

\begin{figure}[htbp]
\centering
\includegraphics[width=\linewidth]{web_casco_bajo_malla.jpg}
\caption{Pestaña del escáner 3D con el casco en procesamiento bajo: malla con la foto como textura y ficha con el tamaño de $V$ y $F$, las medidas
y el tiempo de cálculo.}
\label{fig:visor-bajo}
\end{figure}

\begin{figure}[htbp]
\centering
\includegraphics[width=\linewidth]{web_casco_bajo_puntos.jpg}
\caption{El mismo modelo como nube de puntos (la matriz $V$, sin la matriz $F$).}
\label{fig:visor-puntos}
\end{figure}

\begin{figure}[htbp]
\centering
\begin{minipage}{0.48\linewidth}\centering
\includegraphics[width=\linewidth]{casco_tiempos_etapas.jpg}
\end{minipage}\hfill
\begin{minipage}{0.48\linewidth}\centering
\includegraphics[width=\linewidth]{casco_histograma.jpg}
\end{minipage}
\caption{Izquierda: tiempo de cada etapa para el casco. Derecha: distribución de las profundidades $Z_{\text{frente}}$ de los píxeles del casco
(más cerca de la cámara hacia la izquierda).}
\label{fig:casco-tiempos}
\end{figure}

\begin{figure}[htbp]
\centering
\includegraphics[width=0.8\linewidth]{casco_flujo.jpg}
\caption{Tamaño de cada objeto matemático a lo largo del proceso para el casco: de una matriz de $@h@\times@w@\times3$ números a las matrices
$V$ y $F$.}
\label{fig:casco-flujo}
\end{figure}

\paragraph{Lectura del resultado.} En el visor se reconoce el casco con sus gráficos azules y la visera. Dos defectos son propios del método
con una sola foto: la máscara recorta junto con el casco un pedazo del pupitre sobre el que está apoyado y parte del espaldar verde de la silla (la red los toma como parte del objeto), y
los costados que la cámara no ve quedan estirados, porque la forma se supone redondeada. @alto@
"""
    d = dict(total=n(total / 1000, 1), Nv=N(v["malla"]["N"]), Mf=N(v["malla"]["M"]), tam=" $\\times$ ".join(n(x, 0) for x in v["malla"]["tam_mm"]),
             vol=n(v["malla"]["volumen_mm3"] / 1e6, 1), alg=N(alg), h=v["foto"]["alto"], w=v["foto"]["ancho"],
             alto=("Con procesamiento alto (cuadro~\\ref{tab:alto}) el modelo generativo reconstruye también la parte de atrás y los costados, y el resultado es más completo que "
                   "el de la extrusión (figura~\\ref{fig:visor-alto}); también reproduce, como parte del objeto, el espaldar verde que la máscara dejó pegado al casco.") if alto else "")
    texto = fill(plantilla, d)
    if alto:
        texto += r"""
\begin{figure}[htbp]
\centering
\includegraphics[width=\linewidth]{web_casco_alto.jpg}
\caption{Pestaña del escáner 3D con el casco en procesamiento alto: modelo generativo con textura.}
\label{fig:visor-alto}
\end{figure}
"""
    return texto


# ---------- matrices, vectores e inversas en la metodología --------------------
def metodologia_algebra(v, extra):
    cam, mal = v["camara"], v["malla"]
    K, Ki = np.array(cam["K"]), np.array(cam["K_inv"])
    ej = {e["nombre"]: e for e in v["ejemplos"]}
    A_, B_, C_ = ej["visera"], ej["casco_arriba"], ej["cerca_del_borde"]
    W, H = v["foto"]["ancho"], v["foto"]["alto"]
    Ro = np.diag([1.0, -1.0, -1.0])
    c = np.array(mal["centro_camara"])
    PA = np.array(A_["P"])
    pA = Ro @ (PA - c)                 # a los ejes del objeto
    de_vuelta = Ro.T @ pA + c          # inversa de la rotación y de la traslación
    KP = K @ de_vuelta
    pix = KP / KP[2]
    rA = np.array(A_["rayo"])
    tri = mal["triangulo"]
    rgbA, rgbB = extra["rgbA"], extra["rgbB"]
    plantilla = r"""
\subsection{Matrices, vectores e inversas en el proceso}\label{sec:algebra-proceso}

Todo el proceso del mapeador es álgebra lineal: cada paso es una operación entre matrices y vectores, y cada vez que hay que
deshacer una transformación aparece una inversa. Esta subsección reúne en un solo lugar las matrices, los vectores y las inversas
que se usan, con los valores de la foto del casco; la teoría está en el marco teórico (sección «\nameref{sec:inversa}») y el
desarrollo paso a paso, en la sección «\nameref{sec:casco}».

\paragraph{Matrices del proceso.} El cuadro~\ref{tab:matrices-proceso} muestra las matrices en el orden en que se crean. Los tamaños
cuadran en cada producto: la retroproyección de los $N$ píxeles se hace de una vez con
\begin{equation}
\underbrace{P}_{3\times N}=\underbrace{K^{-1}}_{3\times3}\ \underbrace{H}_{3\times N}\ \underbrace{\operatorname{diag}(Z_1,\dots,Z_N)}_{N\times N},
\end{equation}
donde $(3\times3)(3\times N)$ da $3\times N$ y multiplicar por la diagonal a la derecha escala cada columna. Como en el programa cada punto es una
fila, los vértices quedan en $V\in\mathbb{R}^{N\times3}$ al pasar a los ejes del objeto con la transpuesta de la rotación:
\begin{equation}
V=\bigl(P^{T}-\vect{1}\,\vect{c}^{T}\bigr)\,R_o^{T},\qquad R_o=\operatorname{diag}(1,-1,-1),
\end{equation}
con $\vect{1}$ un vector de unos de longitud $N$ (resta el centro $\vect{c}$ a todas las filas) y $R_o^{T}=R_o$.

\begin{table}[htbp]
\caption{Matrices del proceso (tamaños con la foto del casco)}
\label{tab:matrices-proceso}
\centering\small
\begin{tabular}{@{}p{2.3cm}p{3.1cm}p{5.2cm}p{3.2cm}@{}}
\toprule
Matriz & Tamaño & Qué guarda & Paso \\
\midrule
$I$ & $@h@\times@w@\times3$ & el color de cada píxel & adquisición \\
$M$ & $@h@\times@w@$ & 1 en el objeto, 0 en el fondo & paso 1 \\
$K$ & $3\times3$ & el modelo de la cámara: $f=@f@$, $(c_x,c_y)=(@cx@,@cy@)$ & paso 3 \\
$H$ & $3\times@Nn@$ & los píxeles del casco como columnas $(u,v,1)^{T}$ & paso 3 \\
$\operatorname{diag}(Z)$ & $@Nn@\times@Nn@$ & la profundidad de cada píxel, entre $@zmin@$ y $@zmax@$ mm & paso 3 \\
$P$ & $3\times@Nn@$ & los puntos 3D en los ejes de la cámara & paso 3 \\
$R_o$ & $3\times3$ & giro de $180^\circ$ alrededor de $x$ & paso 3 \\
$\mathcal{M}_o$ & $4\times4$ & $R_o$ y la traslación $-R_o\vect{c}$ juntas (coordenadas homogéneas) & paso 3 \\
$V$ & $@Nv@\times3$ & los vértices de la malla (frente y atrás) & paso 4 \\
$F$ & $@Mf@\times3$ & los índices de los tres vértices de cada triángulo & paso 4 \\
\bottomrule
\end{tabular}
\end{table}

Además de los productos, se usan la \emph{transpuesta} (para pasar de columnas a filas y para invertir rotaciones) y el
\emph{determinante}: el de $K$, $\det K=f^2=@detK@$, comprueba que se puede invertir, y el de la matriz $3\times3$ de los vértices de
cada triángulo da el volumen del tetraedro que forma con el origen.

\paragraph{Vectores del proceso.} Los datos que viajan por el proceso son vectores, y las tres operaciones del curso (norma, producto punto y
producto cruz) aparecen en lugares concretos (cuadro~\ref{tab:vectores-proceso}).

\begin{table}[htbp]
\caption{Vectores del proceso y operaciones que se les aplican (valores del casco)}
\label{tab:vectores-proceso}
\centering\small
\begin{tabular}{@{}p{2.3cm}p{3.0cm}p{4.9cm}p{4.0cm}@{}}
\toprule
Vector & Qué representa & Valor & Operación \\
\midrule
$\vect{c}_A=(r,g,b)$ & color del píxel A & $@cA@$ & norma de la diferencia: $\lVert\vect{c}_A-\vect{c}_B\rVert=@nAB@$ \\
$\vect{p}=(u,v,1)^{T}$ & píxel en coordenadas homogéneas & $(@uA@,@vA@,1)^{T}$ & producto $K^{-1}\vect{p}$ \\
$\vect{r}=K^{-1}\vect{p}$ & dirección del rayo del píxel & $(@r0@,@r1@,1)^{T}$ & norma $@nr@$; ángulo con el eje óptico $@ang@^\circ$ \\
$\vect{P}=Z\,\vect{r}$ & punto 3D (ejes de la cámara) & $(@P0@,@P1@,@P2@)^{T}$ mm & se multiplica por $K$ para volver al píxel \\
$\vect{e}_1,\vect{e}_2$ & dos lados de un triángulo & $\lVert\vect{e}_1\rVert=@ne1@$ & resta de vértices \\
$\vect{n}=\vect{e}_1\times\vect{e}_2$ & normal del triángulo & $(@n0@,@n1@,@n2@)^{T}$ & producto cruz; $\hat n_z=@nu2@>0$ mira al observador \\
\bottomrule
\end{tabular}
\end{table}

El producto punto aparece en el ángulo entre rayos, $\cos\theta=\vect{r}_A\cdot\vect{r}_C/(\lVert\vect{r}_A\rVert\lVert\vect{r}_C\rVert)$, y, en las
habitaciones, en la ecuación del plano $\vect{n}\cdot\vect{P}=d$ y en la intersección rayo--plano $Z=d/(\vect{n}\cdot\vect{r})$.

\paragraph{Matrices de $3\times3$ y de $4\times4$.} Solo las matrices cuadradas se pueden invertir (cuando su determinante no es cero). En el proceso hay
cuatro de $3\times3$ y dos de $4\times4$ (cuadro~\ref{tab:cuadradas}). Se cumple $\det K\cdot\det K^{-1}=f^2\cdot f^{-2}=1$, y $R_o$ y las dos matrices
homogéneas tienen determinante $1$, que es la señal de que no deforman ni cambian volúmenes.

\begin{table}[htbp]
\caption{Matrices cuadradas del proceso, con su determinante y su inversa}
\label{tab:cuadradas}
\centering\small
\begin{tabular}{@{}p{2.2cm}p{1.3cm}p{3.5cm}p{3.5cm}p{3.1cm}@{}}
\toprule
Matriz & Tamaño & Determinante & Inversa & Para qué sirve \\
\midrule
$K$ & $3\times3$ & $f^2=@detK@$ & $K^{-1}$, $\det K^{-1}=@detKi@$ & del píxel al rayo \\
$R_o$ & $3\times3$ & $1$ & $R_o^{-1}=R_o^{T}=R_o$ & girar los ejes \\
$T$ (color) & $3\times3$ & $@detT@$ & $T^{-1}$ & de YCbCr a RGB \\
$[\vect{p}_0;\vect{p}_1;\vect{p}_2]$ & $3\times3$ & $@detTri@$ (de un triángulo) & no se invierte & su determinante da el volumen \\
$\mathcal{M}_o$ & $4\times4$ & $1$ & $\mathcal{M}_o^{-1}=\begin{pmatrix}R_o^{T}&\vect{c}\\\vect{0}^{T}&1\end{pmatrix}$ & cambio de ejes completo \\
$\mathcal{M}$ (alto) & $4\times4$ & $1$ & $\mathcal{M}^{-1}=\mathcal{M}^{T}$ & nodo del modelo generativo \\
\bottomrule
\end{tabular}
\end{table}

\paragraph{Inversas del proceso.} Cada vez que se pasa de un lado a otro hay una matriz y su inversa. El cuadro~\ref{tab:inversas-proceso} reúne las de $3\times3$
y $4\times4$ y dice cómo se obtiene cada una; ninguna necesita un método general porque todas tienen una estructura que la simplifica.

\begin{table}[htbp]
\caption{Inversas de las matrices de $3\times3$ y $4\times4$ y cómo se calculan}
\label{tab:inversas-proceso}
\centering\small
\begin{tabular}{@{}p{2.1cm}p{2.9cm}p{4.2cm}p{4.2cm}@{}}
\toprule
Matriz & Inversa & Cómo se calcula & Para qué sirve \\
\midrule
$K$ ($3\times3$, triangular) & $K^{-1}$ & restar $c_x$, $c_y$ veces la fila 3 y dividir entre $f$ (Gauss--Jordan con 3 operaciones) & del píxel al rayo: $\vect{r}=K^{-1}\vect{p}$ \\
$R_o$ ($3\times3$, ortogonal) & $R_o^{-1}=R_o^{T}$ & trasponer, sin cuentas & volver de los ejes del objeto a los de la cámara \\
$T$ ($3\times3$) & $T^{-1}$ & Gauss--Jordan & pasar de YCbCr a RGB \\
$\mathcal{M}_o$ ($4\times4$) & $\mathcal{M}_o^{-1}$ & por bloques: $R_o^{T}$ y $\vect{c}$ & volver del objeto a la cámara con una sola matriz \\
$\mathcal{M}$ ($4\times4$, alto) & $\mathcal{M}^{-1}=\mathcal{M}^{T}$ (rotación pura) & por bloques, $s^{-1}R^{T}$ y $-s^{-1}R^{T}\vect{t}$ & transformación de nodo del modelo generativo \\
\bottomrule
\end{tabular}
\end{table}

También son cuadradas e invertibles la matriz de la DCT del JPEG ($C$, de $8\times8$, con $C^{-1}=C^{T}$) y $\operatorname{diag}(Z)$ (de $N\times N$, con inversa
$\operatorname{diag}(1/Z)$), pero son matrices grandes que se invierten sin hacer cuentas, por eso no están en el cuadro.

\paragraph{La inversa de la cadena completa.} El paso del píxel al vértice es una cadena de transformaciones, y la inversa de una cadena es la cadena de
las inversas en orden contrario, $(AB)^{-1}=B^{-1}A^{-1}$. Para el píxel A, ir del píxel al vértice es $\vect{p}\ \to\ \vect{P}=Z\,K^{-1}\vect{p}\ \to\
\vect{p}_{\text{obj}}=R_o(\vect{P}-\vect{c})$, y volver es
\begin{equation}
\vect{P}=R_o^{T}\,\vect{p}_{\text{obj}}+\vect{c}=R_o^{T}@pAobj@+@cc@=@Pvuelta@,\qquad
K\vect{P}=@KPv@=Z\,\vect{p},
\end{equation}
y al dividir entre $Z=@Zf@$ se recupera el píxel $(@pixu@,@pixv@,1)$ de partida (con la matriz de $4\times4$ $\mathcal{M}_o$, el mismo viaje de vuelta es un solo
producto, $\mathcal{M}_o^{-1}(\vect{p}_{\text{obj}},1)$): la cadena y su inversa se cancelan, que es lo que
comprueban las pruebas automáticas del servidor ($KK^{-1}=I$ y proyectar un punto retroproyectado devuelve el mismo píxel).

\paragraph{La matriz que no se puede invertir.} La única matriz del proceso sin inversa es la de la proyección, $A=\begin{pmatrix}f&0&c_x-u\\0&f&c_y-v\end{pmatrix}$,
que es de $2\times3$: no es cuadrada, tiene rango $2$ y su espacio nulo es la recta del rayo (sección «\nameref{sec:casco-retro}»). Por eso una sola foto no
alcanza para recuperar la profundidad y hay que agregar información: la silueta y la red de profundidad en los objetos, y los planos del cuarto en las
habitaciones, donde $Z=d/(\vect{n}\cdot\vect{r})$ convierte el sistema en una ecuación con una sola incógnita.
"""
    d = dict(
        h=H, w=W, f=n(cam["f"], 3), cx=n(cam["cx"], 0), cy=n(cam["cy"], 0), Nn=N(mal["n_frente"]), Nv=N(mal["N"]), Mf=N(mal["M"]),
        zmin=n(v["profundidad"]["Zfrente_min"], 0), zmax=n(v["profundidad"]["Zfrente_max"], 0),
        detK=N(int(cam["det_K"])) + "{,}" + f"{cam['det_K'] % 1:.1f}"[2:],
        detKi=sci(1 / cam["det_K"], 3), detT=n(float(np.linalg.det(np.array(extra["T"]))), 4), detTri=n(tri["det"], 3),
        cA="(" + r",\," .join(str(x) for x in rgbA) + ")", nAB=n(extra["normAB"], 1),
        uA=A_["u"], vA=A_["v"], r0=n(rA[0], 4), r1=n(rA[1], 4), nr=n(float(np.linalg.norm(rA)), 5),
        ang=n(float(np.degrees(np.arctan(np.hypot(rA[0], rA[1])))), 2),
        P0=n(PA[0], 2), P1=n(PA[1], 2), P2=n(PA[2], 2),
        ne1=n(float(np.linalg.norm(tri["e1"])), 4), n0=n(tri["normal"][0], 4), n1=n(tri["normal"][1], 4), n2=n(tri["normal"][2], 4),
        nu2=n(tri["normal_unitaria"][2], 4),
        pAobj=vec(pA, 3), cc=vec(c, 3), Pvuelta=vec(de_vuelta, 3), KPv=pm([[nm(x, 1)] for x in KP]), Zf=n(A_["Z_frente"], 2),
        pixu=int(round(pix[0])), pixv=int(round(pix[1])),
    )
    return fill(plantilla, d)


# ---------- principal -----------------------------------------------------------
def main(entrada: str, salida: str, alto_json: str | None) -> None:
    ent, sal = Path(entrada), Path(salida)
    v = json.loads((ent / "valores.json").read_text())
    A = np.load(ent / "arreglos.npz")
    foto = A["foto"]
    mascara = A["mascara"]
    gris = 0.299 * foto[:, :, 2] + 0.587 * foto[:, :, 1] + 0.114 * foto[:, :, 0]
    # bloque 8×8 de mayor varianza dentro del casco
    mejor, bu, bv = 0, 0, 0
    for vv in range(190, 960, 8):
        for uu in range(200, 744, 8):
            if mascara[vv:vv + 8, uu:uu + 8].all() and gris[vv:vv + 8, uu:uu + 8].var() > mejor:
                mejor, bu, bv = gris[vv:vv + 8, uu:uu + 8].var(), uu, vv
    X = np.round(gris[bv:bv + 8, bu:bu + 8]).astype(int)
    C = np.array([[(np.sqrt(1 / 8) if k == 0 else 0.5) * np.cos((2 * j + 1) * k * np.pi / 16) for j in range(8)] for k in range(8)])
    Y = C @ (X - 128) @ C.T
    T = np.array([[0.299, 0.587, 0.114], [-0.168736, -0.331264, 0.5], [0.5, -0.418688, -0.081312]])
    Ti = np.linalg.inv(T)
    ej = {e["nombre"]: e for e in v["ejemplos"]}
    rgb = lambda e: [int(e["bgr"][2]), int(e["bgr"][1]), int(e["bgr"][0])]  # noqa: E731
    rgbA, rgbB, rgbC = rgb(ej["visera"]), rgb(ej["casco_arriba"]), rgb(ej["cerca_del_borde"])
    yc = T @ np.array(rgbB, float) + np.array([0, 128, 128])
    rec = Ti @ (yc - np.array([0, 128, 128]))
    extra = dict(
        w=v["foto"]["ancho"], h=v["foto"]["alto"], w0=v["original"]["ancho"], h0=v["original"]["alto"], bu=bu, bv=bv, X=X, Yr=np.round(Y).astype(int),
        errdct=float(np.abs(C.T @ Y @ C + 128 - X).max()), errC=float(np.abs(C @ C.T - np.eye(8)).max()), T=T.tolist(), Ti=Ti.tolist(),
        rgbA=rgbA, rgbB=rgbB, rgbC=rgbC, ycB=yc.tolist(), rgbBrec=rec.tolist(),
        pA=(ej["visera"]["u"], ej["visera"]["v"]), pB=(ej["casco_arriba"]["u"], ej["casco_arriba"]["v"]), pC=(ej["cerca_del_borde"]["u"], ej["cerca_del_borde"]["v"]),
        normAB=float(np.linalg.norm(np.array(rgbA, float) - np.array(rgbB, float))),
    )
    alto = json.loads(Path(alto_json).read_text()) if alto_json and Path(alto_json).exists() else None
    (sal / "construccion.tex").write_text(construccion(v, extra), encoding="utf8")
    (sal / "marco_inversas.tex").write_text(marco_inversas(), encoding="utf8")
    (sal / "metodologia_algebra.tex").write_text(metodologia_algebra(v, extra), encoding="utf8")
    (sal / "casco_matematica.tex").write_text(casco_matematica(v, extra, ""), encoding="utf8")
    (sal / "casco_alto.tex").write_text(alto_matematica(alto, v), encoding="utf8")
    (sal / "casco_resultados.tex").write_text(casco_resultados(v, alto), encoding="utf8")
    np.save(ent / "bloque_dct.npy", np.stack([X, np.round(Y).astype(int)]))
    print("escrito en", sal, "· bloque DCT en", (bu, bv))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else None)
