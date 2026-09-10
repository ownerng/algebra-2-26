"""
Estudio de ablacion: tres barridos, sus baselines y las figuras del informe.

Regla del proyecto: los experimentos escriben CSV y las figuras se generan
DESDE el CSV, nunca reentrenando. Asi la figura de un informe siempre
corresponde a numeros que existen en disco y se puede regenerar en cualquier
maquina sin repetir el entrenamiento.

    python -m src.experiments.ablation                 # barridos + figuras
    python -m src.experiments.ablation --sweep k
    python -m src.experiments.ablation --figures       # solo redibujar

Las mismas funciones constructoras de figuras las reutiliza la ventana de
analisis de la aplicacion (src/ui/analysis.py), para que la grafica en
pantalla y la del informe sean literalmente la misma.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import config  # noqa: E402
from src.data.loader import load_dataset, stratified_kfold, stratified_split  # noqa: E402
from src.model import lstsq  # noqa: E402
from src.model.knn import KNN  # noqa: E402
from src.model.metrics import accuracy, confusion_matrix, macro_f1  # noqa: E402
from src.pipeline import (apply_representation, compute_raw, fit_representation,  # noqa: E402
                          predict_via, train_via)
from src.ui import theme  # noqa: E402

# Los resultados se guardan por dataset para que correr COIL-100 no borre los
# de Fruits-360. Las figuras, en cambio, viven en docs/figuras con nombre fijo:
# son las que cita el informe, y corresponden al ultimo dataset ejecutado.
DATASET = config.DEFAULT_DATASET


def _ruta(nombre: str) -> Path:
    carpeta = config.RESULTS_DIR / DATASET
    carpeta.mkdir(parents=True, exist_ok=True)
    return carpeta / nombre


def csv_k() -> Path:
    return _ruta("ablacion_k.csv")


def csv_lambda() -> Path:
    return _ruta("ablacion_lambda.csv")


def csv_via() -> Path:
    return _ruta("ablacion_via.csv")


def csv_baselines() -> Path:
    return _ruta("baselines.csv")


def csv_scatter() -> Path:
    return _ruta("scatter_pca.csv")


def npz_eigen() -> Path:
    return _ruta("eigenobjetos.npz")


def csv_confusion(via: str) -> Path:
    return _ruta("confusion_%s.csv" % via)


def _escribir(ruta: Path, cabecera: list[str], filas: list[list]) -> None:
    with open(ruta, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(cabecera)
        w.writerows(filas)
    print("  -> %s (%d filas)" % (ruta.name, len(filas)))


def _leer(ruta: Path) -> tuple[list[str], list[dict]]:
    with open(ruta, newline="", encoding="utf-8") as fh:
        r = csv.DictReader(fh)
        filas = list(r)
    return (r.fieldnames or []), filas


# --------------------------------------------------------------------------
# Barrido 1: accuracy vs k (componentes retenidas), vias B y C
# --------------------------------------------------------------------------

def sweep_k(raw: dict, y: np.ndarray, tr: np.ndarray, te: np.ndarray,
            n_clases: int) -> None:
    filas = []
    for via in ("B", "C"):
        if via not in raw:
            continue
        for k in config.ABLATION_K:
            if k >= len(tr):
                continue
            modelo = train_via(via, raw[via], y, tr, k=k,
                               lam=config.LAMBDA_DEFAULT, n_clases=n_clases)
            pred, _ = predict_via(modelo, raw[via][te])
            acc = accuracy(y[te], pred)

            if via == "B":
                var = float(modelo["eigen"]["explained_var_ratio"].sum())
            else:
                var = float(modelo["pca"].cumulative_variance)

            filas.append([via, k, round(acc, 6), round(var, 6)])
            print("  k=%3d via %s  acc=%.4f  var=%.4f" % (k, via, acc, var))

    _escribir(csv_k(), ["via", "k", "accuracy", "varianza_explicada"], filas)


# --------------------------------------------------------------------------
# Barrido 2: accuracy vs lambda, con numero de condicion
# --------------------------------------------------------------------------

def sweep_lambda(raw: dict, y: np.ndarray, tr: np.ndarray, te: np.ndarray,
                 n_clases: int) -> None:
    filas = []
    for via in config.VIAS:
        if via not in raw:
            continue
        # La representacion no depende de lambda: se ajusta una sola vez y se
        # reutiliza en todo el barrido.
        rep = fit_representation(via, raw[via], tr, config.K_DEFAULT)
        Xtr = apply_representation(rep, raw[via][tr])
        Xte = apply_representation(rep, raw[via][te])
        Y = lstsq.one_hot(y[tr], n_clases)

        for lam in config.ABLATION_LAMBDA:
            W = lstsq.fit_least_squares(Xtr, Y, lam)
            acc = accuracy(y[te], lstsq.predict(Xte, W))
            cond = lstsq.condition_number(Xtr, lam)
            filas.append([via, lam, round(acc, 6), cond])
            print("  lambda=%-8g via %s  acc=%.4f  cond=%.3e" % (lam, via, acc, cond))

    _escribir(csv_lambda(), ["via", "lambda", "accuracy", "cond_XtX"], filas)


# --------------------------------------------------------------------------
# Barrido 3: accuracy vs via, validacion cruzada de 5 pliegues
# --------------------------------------------------------------------------

def sweep_via(raw: dict, y: np.ndarray, n_clases: int) -> None:
    particiones = stratified_kfold(y, config.CV_FOLDS)
    filas = []
    for via in config.VIAS:
        if via not in raw:
            continue
        accs, f1s = [], []
        for tr, te in particiones:
            modelo = train_via(via, raw[via], y, tr, config.K_DEFAULT,
                               config.LAMBDA_DEFAULT, n_clases)
            pred, _ = predict_via(modelo, raw[via][te])
            C = confusion_matrix(y[te], pred, n_clases)
            accs.append(accuracy(y[te], pred))
            f1s.append(macro_f1(C))
        filas.append([via, config.VIA_NOMBRES[via],
                      round(float(np.mean(accs)), 6),
                      round(float(np.std(accs)), 6),
                      round(float(np.mean(f1s)), 6),
                      config.CV_FOLDS])
        print("  via %s  acc=%.4f +- %.4f" % (via, np.mean(accs), np.std(accs)))

    _escribir(csv_via(), ["via", "nombre", "accuracy_media", "accuracy_std",
                        "macro_f1_media", "folds"], filas)


# --------------------------------------------------------------------------
# Baselines
# --------------------------------------------------------------------------

def _baseline_mobilenet(raw: dict, y: np.ndarray, tr: np.ndarray,
                        te: np.ndarray) -> float | None:
    """Techo del modelo preentrenado tal como viene, con su cabeza original.

    La cabeza de MobileNetV3 predice las 1000 clases de ImageNet, no las 10 de
    este trabajo. Para poder medirla se le asigna a cada clase de ImageNet la
    clase del proyecto mas frecuente entre las muestras de entrenamiento que
    la activan (voto mayoritario); las clases de ImageNet no vistas en
    entrenamiento caen a la clase mayoritaria global. No se entrena nada: la
    asignacion es una tabla de conteo sobre salidas congeladas.
    """
    if "C_logits" not in raw:
        return None
    arg = np.argmax(raw["C_logits"], axis=1)
    tabla: dict[int, int] = {}
    for clase_imagenet in np.unique(arg[tr]):
        sel = tr[arg[tr] == clase_imagenet]
        tabla[int(clase_imagenet)] = int(np.bincount(y[sel]).argmax())
    respaldo = int(np.bincount(y[tr]).argmax())
    pred = np.array([tabla.get(int(a), respaldo) for a in arg[te]])
    return accuracy(y[te], pred)


def baselines(raw: dict, y: np.ndarray, tr: np.ndarray, te: np.ndarray,
              n_clases: int) -> None:
    filas = [["aleatorio", "-", round(1.0 / n_clases, 6),
              "piso absoluto, 1/n_clases"]]

    for via in config.VIAS:
        if via not in raw:
            continue
        rep = fit_representation(via, raw[via], tr, config.K_DEFAULT)
        Xtr = apply_representation(rep, raw[via][tr])
        Xte = apply_representation(rep, raw[via][te])

        acc_knn = accuracy(y[te], KNN(config.KNN_K).fit(Xtr, y[tr]).predict(Xte))
        filas.append(["knn_k%d" % config.KNN_K, via, round(acc_knn, 6),
                      "memorizacion sobre la misma representacion"])

        W = lstsq.fit_least_squares(Xtr, lstsq.one_hot(y[tr], n_clases),
                                    config.LAMBDA_DEFAULT)
        acc_ls = accuracy(y[te], lstsq.predict(Xte, W))
        filas.append(["minimos_cuadrados", via, round(acc_ls, 6),
                      "clasificador propio"])

    acc_mn = _baseline_mobilenet(raw, y, tr, te)
    if acc_mn is not None:
        filas.append(["mobilenetv3_cabeza_original", "C", round(acc_mn, 6),
                      "techo del modelo preentrenado sin reemplazar la cabeza"])

    _escribir(csv_baselines(), ["baseline", "via", "accuracy", "nota"], filas)


# --------------------------------------------------------------------------
# Datos de las figuras cualitativas (tambien a traves de CSV / npz)
# --------------------------------------------------------------------------

def datos_figuras(raw: dict, datos: dict, tr: np.ndarray, te: np.ndarray) -> None:
    y, clases = datos["y"], datos["clases"]
    n_clases = len(clases)

    for via in config.VIAS:
        if via not in raw:
            continue
        modelo = train_via(via, raw[via], y, tr, config.K_DEFAULT,
                           config.LAMBDA_DEFAULT, n_clases)
        pred, _ = predict_via(modelo, raw[via][te])
        C = confusion_matrix(y[te], pred, n_clases)
        _escribir(csv_confusion(via), ["real"] + list(clases),
                  [[clases[i]] + [int(v) for v in fila] for i, fila in enumerate(C)])

    if "B" in raw:
        from src.features import eigen as eigmod

        rep = fit_representation("B", raw["B"], tr, 2)
        Z = eigmod.project(raw["B"], rep["eigen"])
        _escribir(csv_scatter(), ["pc1", "pc2", "clase"],
                  [[round(float(a), 6), round(float(b), 6), clases[int(c)]]
                   for (a, b), c in zip(Z[:, :2], y)])

        modelo = eigmod.fit_eigenobjects(raw["B"][tr], 16)
        np.savez_compressed(npz_eigen(), mean=modelo["mean"],
                            components=modelo["components"],
                            ratio=modelo["explained_var_ratio"])
        print("  -> %s" % npz_eigen().name)


# --------------------------------------------------------------------------
# Constructores de figuras: leen CSV, no reentrenan
#
# Cada uno recibe un objeto Figure ya creado (para poder dibujar tanto sobre
# un lienzo de Qt como sobre uno de Agg) y devuelve True si dibujo algo.
# --------------------------------------------------------------------------

def fig_ablacion_k(fig) -> bool:
    if not csv_k().exists():
        return False
    _, filas = _leer(csv_k())
    ax, ax2 = fig.subplots(1, 2)
    for via in ("B", "C"):
        d = [f for f in filas if f["via"] == via]
        if not d:
            continue
        ks = [int(f["k"]) for f in d]
        ax.plot(ks, [float(f["accuracy"]) for f in d], marker="o",
                color=theme.VIA_COLORS[via],
                label="via %s - %s" % (via, config.VIA_NOMBRES[via]))
        ax2.plot(ks, [float(f["varianza_explicada"]) for f in d], marker="o",
                 color=theme.VIA_COLORS[via])
    ax.set_xlabel("componentes retenidas k")
    ax.set_ylabel("exactitud")
    ax2.set_xlabel("componentes retenidas k")
    ax2.set_ylabel("varianza explicada acumulada")
    ax2.axhline(0.90, color=theme.ACCENT, ls="--", lw=1)
    ax.legend()
    return True


def fig_ablacion_lambda(fig) -> bool:
    if not csv_lambda().exists():
        return False
    _, filas = _leer(csv_lambda())
    ax, ax2 = fig.subplots(1, 2)
    for via in config.VIAS:
        d = [f for f in filas if f["via"] == via]
        if not d:
            continue
        # lambda = 0 no existe en escala logaritmica: se dibuja en 1e-6.
        lam = [max(float(f["lambda"]), 1e-6) for f in d]
        ax.plot(lam, [float(f["accuracy"]) for f in d], marker="o",
                color=theme.VIA_COLORS[via], label="via %s" % via)
        ax2.plot(lam, [float(f["cond_XtX"]) for f in d], marker="o",
                 color=theme.VIA_COLORS[via])
    ax.set_xscale("log")
    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax.set_xlabel("lambda (0 dibujado en 1e-6)")
    ax.set_ylabel("exactitud")
    ax2.set_xlabel("lambda")
    ax2.set_ylabel("cond(XtX + lambda I)")
    ax.legend()
    return True


def fig_ablacion_via(fig) -> bool:
    if not csv_via().exists():
        return False
    _, filas = _leer(csv_via())
    ax = fig.subplots()
    x = np.arange(len(filas))
    ax.bar(x, [float(f["accuracy_media"]) for f in filas],
           yerr=[float(f["accuracy_std"]) for f in filas],
           color=[theme.VIA_COLORS[f["via"]] for f in filas],
           width=0.6, capsize=4, ecolor=theme.TEXT_MID)
    ax.set_xticks(x)
    ax.set_xticklabels(["%s\n%s" % (f["via"], f["nombre"]) for f in filas])
    ax.set_ylabel("exactitud (media +- desv., %d folds)" % config.CV_FOLDS)
    ax.set_ylim(0, 1.05)
    return True


def fig_confusion(fig, via: str) -> bool:
    ruta = csv_confusion(via)
    if not ruta.exists():
        return False
    cabecera, filas = _leer(ruta)
    clases = cabecera[1:]
    C = np.array([[int(f[c]) for c in clases] for f in filas], float)
    Cn = C / np.maximum(C.sum(axis=1, keepdims=True), 1)

    ax = fig.subplots()
    ax.imshow(Cn, cmap="magma", vmin=0, vmax=1)
    ax.set_xticks(range(len(clases)))
    ax.set_xticklabels(clases, rotation=90, fontsize=6)
    ax.set_yticks(range(len(clases)))
    ax.set_yticklabels(clases, fontsize=6)
    ax.set_xlabel("prediccion")
    ax.set_ylabel("clase real")
    ax.set_title("via %s - %s" % (via, config.VIA_NOMBRES[via]),
                 color=theme.VIA_COLORS[via])
    ax.grid(False)
    return True


def fig_scatter(fig) -> bool:
    if not csv_scatter().exists():
        return False
    import matplotlib.pyplot as plt

    _, filas = _leer(csv_scatter())
    clases = sorted({f["clase"] for f in filas})
    cmap = plt.get_cmap("turbo", len(clases))
    ax = fig.subplots()
    for i, clase in enumerate(clases):
        d = [f for f in filas if f["clase"] == clase]
        ax.scatter([float(f["pc1"]) for f in d], [float(f["pc2"]) for f in d],
                   s=5, alpha=0.7, color=cmap(i), label=clase, linewidths=0)
    ax.set_xlabel("componente 1")
    ax.set_ylabel("componente 2")
    ax.legend(fontsize=5, markerscale=2, ncol=2)
    return True


def fig_eigenobjetos(fig) -> bool:
    if not npz_eigen().exists():
        return False
    with np.load(npz_eigen()) as z:
        comp, ratio, media = z["components"], z["ratio"], z["mean"]
    lado = int(round(np.sqrt(media.size)))
    ejes = fig.subplots(4, 4)
    for i, ax in enumerate(ejes.ravel()):
        ax.imshow(comp[:, i].reshape(lado, lado), cmap="gray")
        ax.set_title("%d  %.1f%%" % (i + 1, 100 * ratio[i]), fontsize=6)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
    return True


# (nombre de archivo, titulo para la interfaz, constructor, tamano en pulgadas)
FIGURAS = (
    ("ablacion_k.png", "Accuracy vs k", fig_ablacion_k, (8.2, 3.2)),
    ("ablacion_lambda.png", "Accuracy vs lambda", fig_ablacion_lambda, (8.2, 3.2)),
    ("ablacion_via.png", "Accuracy por via", fig_ablacion_via, (4.6, 3.2)),
    ("confusion_A.png", "Confusion A", lambda f: fig_confusion(f, "A"), (4.8, 4.2)),
    ("confusion_B.png", "Confusion B", lambda f: fig_confusion(f, "B"), (4.8, 4.2)),
    ("confusion_C.png", "Confusion C", lambda f: fig_confusion(f, "C"), (4.8, 4.2)),
    ("scatter_pca.png", "Scatter PCA", fig_scatter, (4.8, 4.2)),
    ("eigenobjetos.png", "Eigen-objetos", fig_eigenobjetos, (4.6, 4.8)),
)


def figuras() -> None:
    """Regenera todas las figuras del informe a partir de los CSV."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    theme.apply_matplotlib(plt)
    hechas = 0
    for nombre, _titulo, constructor, tam in FIGURAS:
        fig = plt.figure(figsize=tam)
        if constructor(fig):
            fig.tight_layout()
            fig.savefig(config.FIGURES_DIR / nombre, bbox_inches="tight")
            print("  -> figuras/%s" % nombre)
            hechas += 1
        plt.close(fig)
    print("figuras generadas: %d" % hechas)


# --------------------------------------------------------------------------
# Tablas LaTeX generadas desde los CSV
# --------------------------------------------------------------------------

def tablas() -> None:
    """Convierte los CSV de resultados en tablas .tex con formato APA."""
    if csv_via().exists():
        _, filas = _leer(csv_via())
        cuerpo = "\n".join(
            "%s & %s & %.3f & %.3f & %.3f \\\\" % (
                f["via"], f["nombre"], float(f["accuracy_media"]),
                float(f["accuracy_std"]), float(f["macro_f1_media"]))
            for f in filas)
        _tex("resultados_via.tex", r"""\begin{table}[htbp]
\caption{Exactitud por representacion bajo validacion cruzada}
\label{tab:resultados-via}
\begin{tabular}{llrrr}
\toprule
Via & Representacion & $M$ exactitud & $DE$ & $M$ macro-F1 \\
\midrule
%s
\bottomrule
\end{tabular}
\note{Validacion cruzada estratificada de %d pliegues sobre el mismo
clasificador de minimos cuadrados con $\lambda = %g$ y $k = %d$.
Elaboracion propia.}
\end{table}
""" % (cuerpo, config.CV_FOLDS, config.LAMBDA_DEFAULT, config.K_DEFAULT))

    if csv_baselines().exists():
        _, filas = _leer(csv_baselines())
        cuerpo = "\n".join(
            "%s & %s & %.3f \\\\" % (f["baseline"].replace("_", r"\_"),
                                     f["via"], float(f["accuracy"]))
            for f in filas)
        _tex("baselines.tex", r"""\begin{table}[htbp]
\caption{Comparacion contra los baselines}
\label{tab:baselines}
\begin{tabular}{llr}
\toprule
Baseline & Via & Exactitud \\
\midrule
%s
\bottomrule
\end{tabular}
\note{Todos los valores sobre la misma particion de prueba.
Elaboracion propia.}
\end{table}
""" % cuerpo)


def _tex(nombre: str, contenido: str) -> None:
    ruta = config.TABLES_DIR / nombre
    ruta.write_text(contenido, encoding="utf-8")
    print("  -> tablas/%s" % nombre)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="Estudio de ablacion de SCOAL.")
    ap.add_argument("--dataset", choices=config.DATASETS,
                    default=config.DEFAULT_DATASET)
    ap.add_argument("--sweep", choices=("all", "k", "lambda", "via", "baselines"),
                    default="all")
    ap.add_argument("--figures", action="store_true",
                    help="solo redibujar desde los CSV existentes")
    args = ap.parse_args()

    # Los CSV de este dataset viven en results/<dataset>/.
    global DATASET
    DATASET = args.dataset

    if args.figures:
        figuras()
        tablas()
        return

    datos = load_dataset(args.dataset)
    y = datos["y"]
    n_clases = len(datos["clases"])
    raw = compute_raw(args.dataset, datos)
    tr, te = stratified_split(y)
    print("dataset %s: %d imagenes, %d clases, vias disponibles: %s"
          % (args.dataset, len(y), n_clases,
             ", ".join(v for v in config.VIAS if v in raw)))

    if args.sweep in ("all", "k"):
        print("barrido 1: accuracy vs k")
        sweep_k(raw, y, tr, te, n_clases)
    if args.sweep in ("all", "lambda"):
        print("barrido 2: accuracy vs lambda")
        sweep_lambda(raw, y, tr, te, n_clases)
    if args.sweep in ("all", "via"):
        print("barrido 3: accuracy vs via (CV %d folds)" % config.CV_FOLDS)
        sweep_via(raw, y, n_clases)
    if args.sweep in ("all", "baselines"):
        print("baselines")
        baselines(raw, y, tr, te, n_clases)
    if args.sweep == "all":
        print("datos de figuras")
        datos_figuras(raw, datos, tr, te)
        figuras()
        tablas()


if __name__ == "__main__":
    main()
