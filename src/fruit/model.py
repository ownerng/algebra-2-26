"""
Arquitecturas de la via D: red preentrenada en ImageNet con cabeza nueva.

Por que estas cuatro y no otras (todas vienen en torchvision, sin dependencias
extra; exactitud top-1 en ImageNet segun la documentacion de torchvision):

  efficientnet_v2_s   84.2 %  21 M parametros   defecto: mejor relacion
                              exactitud / costo en CPU
  convnext_tiny       82.5 %  28 M              la mas robusta a cambio de
                              dominio; buena segunda opcion con GPU
  resnet50            80.9 %  26 M              referencia clasica para el informe
  mobilenet_v3_large  75.3 %  5.5 M             si la CPU del equipo es lenta

Los pesos de ImageNet los baja torchvision la primera vez. Si la red bloquea
download.pytorch.org, ponlos a mano en `data/raw/torch/<arch>.pth`
(ver `pesos_locales`).
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import config

log = logging.getLogger(__name__)

os.environ.setdefault("TORCH_HOME", str(config.RAW_DIR / "torch"))

ARQUITECTURAS = ("efficientnet_v2_s", "convnext_tiny", "resnet50", "mobilenet_v3_large")


def pesos_locales(arch: str) -> Path:
    return config.RAW_DIR / "torch" / ("%s.pth" % arch)


def construir(arch: str, n_clases: int, preentrenado: bool = True, dropout: float = 0.3):
    """Backbone de ImageNet + cabeza lineal nueva de `n_clases` salidas."""
    import torch
    from torch import nn
    from torchvision import models

    if arch not in ARQUITECTURAS:
        raise ValueError("arquitectura desconocida %r; opciones: %s"
                         % (arch, ", ".join(ARQUITECTURAS)))

    constructor = getattr(models, arch)
    local = pesos_locales(arch)
    if preentrenado and local.exists():
        log.info("pesos de %s puestos a mano: %s", arch, local)
        red = constructor(weights=None)
        red.load_state_dict(torch.load(local, map_location="cpu"))
    elif preentrenado:
        try:
            red = constructor(weights="DEFAULT")
        except Exception as exc:
            raise RuntimeError(
                "No se pudieron descargar los pesos preentrenados de %s (%s).\n"
                "Bajalos a mano y guardalos como %s\n"
                "  (state_dict de torchvision; url en la documentacion de "
                "torchvision.models.%s)" % (arch, exc, local, arch)) from exc
    else:
        red = constructor(weights=None)

    if arch == "efficientnet_v2_s":
        n = red.classifier[-1].in_features
        red.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(n, n_clases))
    elif arch == "mobilenet_v3_large":
        n = red.classifier[-1].in_features
        red.classifier[-2] = nn.Dropout(dropout)
        red.classifier[-1] = nn.Linear(n, n_clases)
    elif arch == "convnext_tiny":
        n = red.classifier[-1].in_features
        red.classifier[-1] = nn.Linear(n, n_clases)
    elif arch == "resnet50":
        n = red.fc.in_features
        red.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(n, n_clases))
    return red


def parametros_cabeza(red, arch: str):
    """Parametros de la capa nueva (los que se entrenan con lr alto al inicio)."""
    cabeza = red.fc if arch == "resnet50" else red.classifier
    return list(cabeza.parameters())


def guardar(ruta: Path, red, arch: str, clases: list[str], lado: int,
            temperatura: float, umbrales: dict, metricas: dict) -> Path:
    """Checkpoint autocontenido en fp16 (la mitad de tamano, sin perdida util)."""
    import torch
    ruta.parent.mkdir(parents=True, exist_ok=True)
    estado = {k: (v.half() if v.is_floating_point() else v)
              for k, v in red.state_dict().items()}
    torch.save({
        "arch": arch, "clases": list(clases), "lado": lado,
        "temperatura": float(temperatura), "umbrales": umbrales,
        "metricas": metricas, "state_dict": estado,
    }, ruta)
    log.info("modelo de fruta guardado en %s (%.1f MB)", ruta,
             ruta.stat().st_size / 2**20)
    return ruta


def cargar(ruta: Path = config.FRUIT_MODEL_PATH):
    """Devuelve (red en eval, metadatos). Sin gradientes, en fp32."""
    import torch
    paquete = torch.load(ruta, map_location="cpu", weights_only=False)
    red = construir(paquete["arch"], len(paquete["clases"]), preentrenado=False)
    red.load_state_dict({k: (v.float() if v.is_floating_point() else v)
                         for k, v in paquete["state_dict"].items()})
    for p in red.parameters():
        p.requires_grad = False
    red.eval()
    meta = {k: v for k, v in paquete.items() if k != "state_dict"}
    return red, meta
