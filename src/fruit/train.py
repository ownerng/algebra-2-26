"""
Afinado (fine-tuning) de una CNN preentrenada sobre todo Fruits-360.

    python -m src.fruit.train                       # defecto: efficientnet_v2_s
    python -m src.fruit.train --arch convnext_tiny --epocas 12
    python -m src.fruit.train --real fotos_reales/  # evalua ademas en TUS fotos
    python -m src.fruit.train --rapido              # humo: 2 epocas, 40 img/clase

Receta (en este orden, cada paso se justifica en el informe):

  1. Calentamiento: 1 epoca entrenando solo la cabeza nueva. Sin esto, los
     gradientes de una cabeza aleatoria destruyen las capas preentrenadas.
  2. Afinado completo con tasa de aprendizaje diferenciada (backbone = 10 %
     de la de la cabeza), AdamW, coseno con calentamiento lineal, suavizado
     de etiquetas 0.1 y recorte de gradiente.
  3. Aumentos que rompen el dominio de estudio (fondos, luz, desenfoque, JPEG);
     ver data.py. Es lo que sostiene la exactitud fuera del dataset.
  4. Calibracion por temperatura y umbrales de rechazo en validacion.
  5. Tres evaluaciones separadas, que NO deben mezclarse en el informe:
       - Test oficial de Fruits-360 (dominio de estudio; cercano a 100 %)
       - Test con dominio roto (fondos y degradacion deterministas)
       - Fotos reales propias, si se pasa --real <carpeta/clase/*.jpg>

AVISO de honestidad: Training y Test de Fruits-360 salen de la misma sesion de
fotos de cada fruta (fotogramas de un video), asi que el 99.x % del test oficial
sobreestima el desempeno en el mundo real. La cifra que vale para el proyecto
de grado es la de fotos reales.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import random
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import config  # noqa: E402
from src.fruit import data, infer, model  # noqa: E402

log = logging.getLogger(__name__)


# --------------------------------------------------------------------------
# Utilidades
# --------------------------------------------------------------------------

def _semilla_worker(_id: int) -> None:
    """Cada worker necesita su propia semilla de numpy o todos aumentan igual."""
    import torch
    s = torch.initial_seed() % (2**32 - 1)
    np.random.seed(s)
    random.seed(s)


def _dispositivo():
    import torch
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _cargador(ds, batch: int, shuffle: bool, workers: int):
    from torch.utils.data import DataLoader
    return DataLoader(ds, batch_size=batch, shuffle=shuffle, num_workers=workers,
                      worker_init_fn=_semilla_worker, drop_last=shuffle,
                      pin_memory=False, persistent_workers=workers > 0)


def _logits_de(red, cargador, disp) -> tuple[np.ndarray, np.ndarray]:
    import torch
    red.eval()
    zs, ys = [], []
    with torch.inference_mode():
        for x, y in cargador:
            zs.append(red(x.to(disp)).float().cpu().numpy())
            ys.append(y.numpy())
    return np.concatenate(zs), np.concatenate(ys)


def _submuestrear(rutas, etiquetas, por_clase: int, semilla: int = config.SEED):
    rng = np.random.default_rng(semilla)
    y = np.asarray(etiquetas)
    idx = []
    for c in np.unique(y):
        pos = np.flatnonzero(y == c)
        rng.shuffle(pos)
        idx += pos[:por_clase].tolist()
    idx = sorted(idx)
    return [rutas[i] for i in idx], [etiquetas[i] for i in idx]


# --------------------------------------------------------------------------
# Evaluacion
# --------------------------------------------------------------------------

def reporte(logits: np.ndarray, y: np.ndarray, clases: list[str], T: float,
            umbrales: dict) -> dict:
    """Metricas de una evaluacion: top-1, top-5, F1 macro, calibracion, rechazo."""
    p = infer.softmax(logits, T)
    pred = p.argmax(1)
    n_c = len(clases)
    top5 = np.argsort(-p, axis=1)[:, :min(5, n_c)]
    conf = p.max(1)
    ent = infer.entropia_normalizada(p)
    conocido = infer.es_conocido(conf, ent, umbrales)

    # F1 macro desde la matriz de confusion (sin sklearn: el motor es propio).
    cm = np.zeros((n_c, n_c), np.int64)
    np.add.at(cm, (y, pred), 1)
    tp = np.diag(cm).astype(float)
    prec = tp / np.maximum(cm.sum(0), 1)
    rec = tp / np.maximum(cm.sum(1), 1)
    f1 = 2 * prec * rec / np.maximum(prec + rec, 1e-12)
    presentes = cm.sum(1) > 0

    # ECE: diferencia media entre confianza y exactitud por cajas de confianza.
    cajas = np.linspace(0, 1, 11)
    ece = 0.0
    for a, b in zip(cajas[:-1], cajas[1:]):
        m = (conf > a) & (conf <= b)
        if m.any():
            ece += m.mean() * abs((pred[m] == y[m]).mean() - conf[m].mean())

    acierto = pred == y
    return {
        "n": int(len(y)),
        "top1": float(acierto.mean()),
        "top5": float(np.mean([y[i] in top5[i] for i in range(len(y))])),
        "f1_macro": float(f1[presentes].mean()),
        "ece": float(ece),
        "nll": float(-np.log(np.clip(p[np.arange(len(y)), y], 1e-12, 1)).mean()),
        "rechazo": float(1 - conocido.mean()),
        # exactitud sobre lo que NO se rechaza: lo que ve un usuario en produccion
        "top1_aceptadas": float(acierto[conocido].mean()) if conocido.any() else float("nan"),
        "_confusion": cm,
        "_f1_clase": f1,
    }


def _limpio(rep: dict) -> dict:
    return {k: v for k, v in rep.items() if not k.startswith("_")}


def _confusiones_top(cm: np.ndarray, clases: list[str], n: int = 15) -> list[dict]:
    fuera = cm.copy()
    np.fill_diagonal(fuera, 0)
    flat = np.argsort(-fuera, axis=None)[:n]
    return [{"real": clases[k // len(clases)], "predicho": clases[k % len(clases)],
             "veces": int(fuera.flat[k])}
            for k in flat if fuera.flat[k] > 0]


def _guardar_csv(destino: Path, rep: dict, clases: list[str]) -> None:
    cm = rep["_confusion"]
    with open(destino, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["clase", "soporte", "recall", "f1"])
        for i, c in enumerate(clases):
            s = int(cm[i].sum())
            if s:
                w.writerow([c, s, "%.4f" % (cm[i, i] / s), "%.4f" % rep["_f1_clase"][i]])


def evaluar_carpeta_real(red, disp, clases: list[str], carpeta: Path, lado: int,
                         T: float, umbrales: dict, batch: int) -> dict | None:
    """Fotos reales propias: carpeta/<variedad>/*.jpg. Solo cuentan las variedades
    que el modelo conoce; el resto se lista aparte para no perderlas en silencio."""
    if not carpeta.is_dir():
        log.warning("--real %s no es una carpeta", carpeta)
        return None
    indice = {data.clave(c): i for i, c in enumerate(clases)}
    rutas, etiquetas, ignoradas = [], [], set()
    for ruta, nombre in zip(*data.indexar(carpeta)):
        if data.clave(nombre) in indice:
            rutas.append(ruta)
            etiquetas.append(indice[data.clave(nombre)])
        else:
            ignoradas.add(nombre)
    if not rutas:
        log.warning("--real: ninguna carpeta coincide con una variedad del modelo")
        return None
    ds = data.FrutasDataset(rutas, etiquetas, "eval", lado)
    z, y = _logits_de(red, _cargador(ds, batch, False, 0), disp)
    rep = reporte(z, y, clases, T, umbrales)
    rep["_ignoradas"] = sorted(ignoradas)
    return rep


# --------------------------------------------------------------------------
# Entrenamiento
# --------------------------------------------------------------------------

def entrenar(args) -> dict:
    import torch
    from torch import nn

    random.seed(config.SEED)
    np.random.seed(config.SEED)
    torch.manual_seed(config.SEED)

    raiz = data.buscar_raiz(args.data)
    clases, indice = data.mapa_clases(raiz / "Training", agrupar=not args.sin_agrupar,
                                      solo_frutas=not args.con_verduras)
    log.info("Fruits-360 en %s: %d clases", raiz, len(clases))

    r_tr, n_tr = data.indexar(raiz / "Training")
    r_tr = [r for r, n in zip(r_tr, n_tr) if n in indice]
    y_tr = [indice[n] for n in n_tr if n in indice]
    test_dir = raiz / "Test"
    r_te, n_te = data.indexar(test_dir) if test_dir.is_dir() else ([], [])
    y_te = [indice[n] for n in n_te if n in indice]
    r_te = [r for r, n in zip(r_te, n_te) if n in indice]

    if args.rapido:
        args.epocas, args.por_clase = min(args.epocas, 2), 40
    if args.por_clase:
        r_tr, y_tr = _submuestrear(r_tr, y_tr, args.por_clase)
        if r_te:
            r_te, y_te = _submuestrear(r_te, y_te, max(10, args.por_clase // 3))

    i_tr, i_va = data.partir_validacion(y_tr, 0.05)
    fondos = Path(args.fondos) if args.fondos else None
    ds_tr = data.FrutasDataset([r_tr[i] for i in i_tr], [y_tr[i] for i in i_tr],
                               "train", args.lado, fondos)
    ds_va = data.FrutasDataset([r_tr[i] for i in i_va], [y_tr[i] for i in i_va],
                               "shift", args.lado, fondos)
    log.info("train %d | validacion %d (con dominio roto) | test %d",
             len(ds_tr), len(ds_va), len(r_te))

    disp = _dispositivo()
    red = model.construir(args.arch, len(clases), preentrenado=not args.sin_preentrenar)
    red.to(disp)
    if disp.type == "cuda":
        red = red.to(memory_format=torch.channels_last)
    usar_amp = disp.type == "cuda"
    escalador = torch.amp.GradScaler("cuda", enabled=usar_amp)
    log.info("dispositivo %s, arquitectura %s, %d parametros", disp, args.arch,
             sum(p.numel() for p in red.parameters()))

    cabeza = model.parametros_cabeza(red, args.arch)
    ids_cabeza = {id(p) for p in cabeza}
    cuerpo = [p for p in red.parameters() if id(p) not in ids_cabeza]
    opt = torch.optim.AdamW(
        [{"params": cuerpo, "lr": args.lr * 0.1}, {"params": cabeza, "lr": args.lr}],
        weight_decay=0.05)
    base_lrs = [g["lr"] for g in opt.param_groups]
    criterio = nn.CrossEntropyLoss(label_smoothing=0.1)

    carg_tr = _cargador(ds_tr, args.batch, True, args.workers)
    carg_va = _cargador(ds_va, args.batch, False, args.workers)
    pasos_epoca = len(carg_tr)
    total = pasos_epoca * args.epocas
    calent = pasos_epoca // 2

    def lr_en(paso: int) -> float:
        if paso < calent:
            return (paso + 1) / calent
        t = (paso - calent) / max(1, total - calent)
        return 0.5 * (1 + math.cos(math.pi * t)) * 0.99 + 0.01

    mejor, mejor_estado, paso = -1.0, None, 0
    historial = []
    for epoca in range(args.epocas):
        # Epoca 0: solo la cabeza. Congelar es quitar requires_grad al cuerpo.
        entrenar_cuerpo = epoca >= args.calentamiento
        for p in cuerpo:
            p.requires_grad = entrenar_cuerpo
        red.train()
        t0, perdida_acum, n_acum = time.time(), 0.0, 0
        for x, y in carg_tr:
            for g, base in zip(opt.param_groups, base_lrs):
                g["lr"] = base * lr_en(paso)
            x, y = x.to(disp), y.to(disp)
            if disp.type == "cuda":
                x = x.contiguous(memory_format=torch.channels_last)
            with torch.autocast(device_type=disp.type, enabled=usar_amp):
                perdida = criterio(red(x), y)
            opt.zero_grad(set_to_none=True)
            escalador.scale(perdida).backward()
            escalador.unscale_(opt)
            nn.utils.clip_grad_norm_(red.parameters(), 2.0)
            escalador.step(opt)
            escalador.update()
            perdida_acum += perdida.item() * len(y)
            n_acum += len(y)
            paso += 1
            if args.rapido and paso % 5 == 0:
                log.info("  paso %d/%d perdida %.3f", paso, total, perdida_acum / n_acum)

        z, y = _logits_de(red, carg_va, disp)
        acc = float((z.argmax(1) == y).mean())
        historial.append({"epoca": epoca + 1, "perdida": perdida_acum / max(n_acum, 1),
                          "val_top1_dominio_roto": acc, "segundos": time.time() - t0})
        log.info("epoca %d/%d  perdida %.4f  val(dominio roto) %.4f  %.0fs",
                 epoca + 1, args.epocas, historial[-1]["perdida"], acc,
                 historial[-1]["segundos"])
        print("epoca %d/%d  perdida %.4f  val %.4f" % (
            epoca + 1, args.epocas, historial[-1]["perdida"], acc), flush=True)
        if acc >= mejor:
            mejor = acc
            mejor_estado = {k: v.detach().cpu().clone()
                            for k, v in red.state_dict().items()}

    red.load_state_dict(mejor_estado)
    red.to(disp)

    # -- calibracion y umbrales en validacion ---------------------------
    z_va, y_va = _logits_de(red, carg_va, disp)
    T = infer.ajustar_temperatura(z_va, y_va)
    umbrales = infer.fijar_umbrales(z_va, y_va, T)
    log.info("temperatura %.3f, umbrales %s", T, umbrales)

    # -- evaluaciones (nunca se mezclan en el informe) -------------------
    metricas: dict = {"arch": args.arch, "epocas": args.epocas,
                      "n_clases": len(clases), "temperatura": T, "historial": historial}
    reps: dict[str, dict] = {}
    if r_te:
        ds_te = data.FrutasDataset(r_te, y_te, "eval", args.lado)
        ds_ts = data.FrutasDataset(r_te, y_te, "shift", args.lado, fondos)
        for nombre, ds in (("test_oficial", ds_te), ("test_dominio_roto", ds_ts)):
            z, y = _logits_de(red, _cargador(ds, args.batch, False, args.workers), disp)
            reps[nombre] = reporte(z, y, clases, T, umbrales)
    if args.real:
        rep = evaluar_carpeta_real(red, disp, clases, Path(args.real), args.lado,
                                   T, umbrales, args.batch)
        if rep:
            reps["fotos_reales"] = rep
    for nombre, rep in reps.items():
        metricas[nombre] = _limpio(rep)
        print("%-18s n=%-6d top1 %.4f  top5 %.4f  F1 %.4f  ECE %.4f  rechazo %.3f"
              % (nombre, rep["n"], rep["top1"], rep["top5"], rep["f1_macro"],
                 rep["ece"], rep["rechazo"]))

    # -- salida ----------------------------------------------------------
    salida = Path(args.salida)
    model.guardar(salida, red.cpu(), args.arch, clases, args.lado, T, umbrales,
                  {k: v for k, v in metricas.items() if k != "historial"})
    carpeta = salida.parent
    peor = {}
    for nombre, rep in reps.items():
        _guardar_csv(carpeta / ("por_clase_%s.csv" % nombre), rep, clases)
        np.savetxt(carpeta / ("confusion_%s.csv" % nombre), rep["_confusion"],
                   fmt="%d", delimiter=",")
        peor[nombre] = _confusiones_top(rep["_confusion"], clases)
    metricas["confusiones_top"] = peor
    with open(carpeta / "metricas.json", "w", encoding="utf-8") as fh:
        json.dump(metricas, fh, indent=2, ensure_ascii=False)
    return metricas


def main() -> None:
    ap = argparse.ArgumentParser(description="Afina una CNN preentrenada en Fruits-360.")
    ap.add_argument("--arch", default=config.FRUIT_ARCH, choices=model.ARQUITECTURAS)
    ap.add_argument("--data", default=None, help="carpeta con Training/ y Test/")
    ap.add_argument("--real", default=None, help="carpeta <variedad>/*.jpg con TUS fotos")
    ap.add_argument("--fondos", default=None, help="carpeta con fotos de fondos reales")
    ap.add_argument("--sin-agrupar", action="store_true",
                    help="una clase por carpeta (Apple Golden 1/2/3 separadas)")
    ap.add_argument("--con-verduras", action="store_true",
                    help="incluye verduras, frutos secos y hongos (por defecto: solo frutas)")
    ap.add_argument("--epocas", type=int, default=10)
    ap.add_argument("--calentamiento", type=int, default=1, help="epocas solo-cabeza")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3, help="de la cabeza; el cuerpo usa 10 %%")
    ap.add_argument("--lado", type=int, default=config.FRUIT_IMG)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--por-clase", type=int, default=0, help="limita imagenes/clase (0 = todas)")
    ap.add_argument("--salida", default=str(config.FRUIT_MODEL_PATH))
    ap.add_argument("--rapido", action="store_true", help="prueba de humo (2 epocas, 40 img/clase)")
    ap.add_argument("--sin-preentrenar", action="store_true",
                    help="aleatorio: solo para pruebas sin red")
    args = ap.parse_args()
    entrenar(args)


if __name__ == "__main__":
    main()
