"""Reparte las secciones generadas (casco_matematica, vectores_casco, habitacion_matematica)
en piezas, una por paso del proceso, para que la resolución con números quede dentro de
cada paso de la metodología y no en secciones aparte.

Uso: python repartir_pasos.py GENERADO/ docs/secciones/
  GENERADO/ contiene la salida de casco_latex.py y habitacion_latex.py.
"""
import re
import sys
from pathlib import Path


def piezas_por(texto, patron):
    """Parte `texto` en trozos que empiezan en cada coincidencia de `patron` (multilínea)."""
    cortes = [m.start() for m in re.finditer(patron, texto, flags=re.M)]
    cabeza = texto[:cortes[0]] if cortes else texto
    return cabeza, [texto[a:b] for a, b in zip(cortes, cortes[1:] + [len(texto)])]


def main(gen, sal):
    gen, sal = Path(gen), Path(sal)
    escribir = lambda nombre, t: (sal / nombre).write_text(t.strip() + "\n", encoding="utf8")  # noqa: E731

    # ---------------- casco ----------------
    t = (gen / "casco_matematica.tex").read_text(encoding="utf8")
    # el resumen (cuadro) es lo último; se separa
    i = t.index(r"\paragraph{Resumen: matrices, vectores e inversas del proceso.}")
    resumen, t = t[i:], t[:i]
    _, sub = piezas_por(t, r"^\\subsection\{")
    por = {re.search(r"\\label\{(sec:[^}]+)\}", s).group(1): s for s in sub}
    titulos = {
        "sec:casco-camara": "Resolución del paso 3: la matriz $K$ y su inversa con los valores del casco",
        "sec:casco-mascara": "Resolución del paso 1: la máscara $M$ del casco",
        "sec:casco-profundidad": "Resolución del paso 2: la profundidad del casco, de la disparidad a $Z$",
        "sec:casco-retro": "Resolución del paso 3: de píxeles a puntos 3D",
        "sec:casco-ejes": "Resolución del paso 3: del sistema de la cámara al del objeto",
        "sec:casco-malla": "Resolución del paso 4: la malla $V$, $F$ del casco",
    }
    for lab, tit in titulos.items():
        por[lab] = re.sub(r"^\\subsection\{[^\n]*?\}(\\label)", lambda m: "\\subsubsection{" + tit + "}" + m.group(1), por[lab], count=1, flags=re.M)
    intro = por["sec:casco"].replace(
        r"\subsection{Caso de estudio: el casco de referencia}",
        r"\subsection{La foto de ejemplo: el casco de referencia}")
    intro = intro.replace(
        "Para que la matemática no quede en el aire, todo el desarrollo que sigue se hace con una fotografía real y con",
        "Para que la matemática no quede en el aire, cada paso de la metodología se resuelve a continuación, dentro del mismo paso, con una fotografía real y con")
    escribir("paso0_casco.tex", intro)
    escribir("paso1_casco.tex", por["sec:casco-mascara"])
    escribir("paso2_casco.tex", por["sec:casco-profundidad"])
    retro = por["sec:casco-retro"]
    retro = re.sub(r"\\paragraph\{Vectores de los rayos\.\}.*?(?=\\paragraph\{Muchos)", "", retro, flags=re.S)
    escribir("paso3a_casco.tex", por["sec:casco-camara"] + "\n" + retro)
    escribir("paso3b_casco.tex", por["sec:casco-ejes"])
    escribir("paso4_casco.tex", por["sec:casco-malla"].replace("-0{,}3248", "-0{,}3247"))

    # ---------------- vectores del casco ----------------
    t = (gen / "vectores_casco.tex").read_text(encoding="utf8")
    cab, par = piezas_por(t, r"^\\paragraph\{")
    p = {re.match(r"\\paragraph\{([^.]+)\.", s).group(1): s for s in par}
    intro_v = (r"\paragraph{Los vectores y sus tres operaciones.} Las matrices del proceso se aplican siempre sobre \emph{vectores}, y es con los "
               "vectores con los que se hacen las tres operaciones del curso: la norma $\\lVert\\vect{a}\\rVert$, el producto punto "
               "$\\vect{a}\\cdot\\vect{b}$ y el producto cruz $\\vect{a}\\times\\vect{b}$. Los párrafos siguientes siguen, con los números del casco, "
               "cada vector que aparece en este paso y dicen qué significa cada operación.\n\n")
    px = p["Del píxel al rayo"]
    px = px[px.index("La \\emph{norma}"):]
    px = "\\paragraph{Norma y ángulo de un rayo.} " + px
    pt = p["Del rayo al punto"]
    pt = ("\\paragraph{Longitud del punto.} El punto $\\vect{P}_A$ se calculó arriba. Su norma, $\\lVert\\vect{P}_A\\rVert=301{,}97$ mm, es la distancia "
          "real de la cámara a ese punto, mayor que $Z$ porque el rayo está inclinado ($\\lVert\\vect{P}\\rVert=Z\\lVert\\vect{r}\\rVert$).\n\n")
    escribir("paso1_vectores.tex", p["El color como vector"])
    escribir("paso3a_vectores.tex", intro_v + px + pt + p["Producto punto entre dos rayos"])
    ob = p["Del punto de la cámara al objeto"]
    ob = ob[ob.index("\\begin{table}"):]
    escribir("paso3b_vectores.tex",
             "\\paragraph{Los tres puntos en los dos sistemas.} El cuadro~\\ref{tab:objeto-vec} aplica $\\vect{p}=R_o(\\vect{P}-\\vect{c})$ a los puntos A, B y C: "
             "restar $\\vect{c}$ lleva el origen al centro del casco y $R_o$ cambia el signo de $y$ y de $z$, sin cambiar la norma.\n\n" + ob)
    tr = p["Los vectores de un triángulo y el producto cruz"]
    tr = ("\\paragraph{El producto cruz componente por componente.} Los lados $\\vect{e}_1$ y $\\vect{e}_2$ del triángulo anterior son vectores, y su "
          "producto cruz $\\vect{n}=\\vect{e}_1\\times\\vect{e}_2$ es perpendicular al triángulo. Sus componentes (figura~\\ref{fig:vec-triangulo}) salen así:\n") + tr[tr.index("\\begin{align*}"):]
    escribir("paso4_vectores.tex", tr)

    # ---------------- sala ----------------
    t = (gen / "habitacion_matematica.tex").read_text(encoding="utf8").replace("sec:vectores-casco", "sec:casco-retro")
    cab, par = piezas_por(t, r"^\\paragraph\{")
    # cabecera: subsection + figuras -> subsubsection
    cab = cab.replace(r"\subsection{Caso de estudio: una sala, paso a paso}", r"\subsubsection{La sala de ejemplo}")
    cab = cab.replace("Igual que con el casco, el proceso de las habitaciones se desarrolla aquí con",
                      "Igual que con el casco, cada paso de la reconstrucción se resuelve a continuación, dentro del mismo paso, con")
    cab = cab.replace("En la figura~\\ref{fig:hab-flujo} están los seis pasos", "En la figura~\\ref{fig:hab-flujo} están los seis pasos")
    nombres = [re.match(r"\\paragraph\{([^.]+)\.", s).group(1) for s in par]
    grupo = {1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 0: []}
    actual = 0
    for nom, s in zip(nombres, par):
        m = re.match(r"Paso (\d)", nom)
        if m:
            actual = int(m.group(1))
        s = re.sub(r"^\\paragraph\{Paso \d: ", r"\\paragraph{Con los números de la sala: ", s, count=1)
        grupo[actual].append(s)
    escribir("paso0_sala.tex", cab + "\n" + "".join(grupo[0]))
    for k in range(1, 7):
        escribir(f"paso{k}_sala.tex", "".join(grupo[k]))
    print("repartido en", sal)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
