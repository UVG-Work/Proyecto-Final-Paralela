"""FASE 1 - preparacion de datos (CPU). Todo lo que toca archivos e imagenes.

Este modulo NO importa torch, a proposito: en Windows multiprocessing arranca
los procesos hijos con 'spawn' y cada hijo importa el modulo de la funcion que
ejecuta. Si cargar_imagen viviera junto a 'import torch', cada hijo cargaria
PyTorch + las DLL de CUDA (segundos y cientos de MB por proceso).

Lo comparten secuencial.py y la version optimizada: si se cambia algo aqui,
cambia para ambas y la comparacion sigue siendo justa.
"""
import csv
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image

RAIZ = Path(__file__).resolve().parent
SPLITS_CSV = RAIZ / "datos" / "dataset_splits.csv"
TIEMPOS_CSV = RAIZ / "resultados" / "tiempos.csv"

TAM = 64  # 200x200 -> 64x64 (metodologia del Lab 3, seccion 9.2)

COLUMNAS = ["fecha", "version", "dispositivo", "hilos", "epocas",
            "corrida", "fase", "segundos", "detalle"]


def ruta_dataset(ruta=None):
    """Carpeta con una subcarpeta por clase (A..Z, del, nothing, space).

    Sin argumento usa kagglehub, que devuelve la copia en cache si ya existe.
    """
    if ruta is None:
        import kagglehub
        ruta = kagglehub.dataset_download("grassknoted/asl-alphabet")
    ruta = Path(ruta)
    # El zip de Kaggle anida la carpeta dos veces.
    for candidata in (ruta / "asl_alphabet_train" / "asl_alphabet_train",
                      ruta / "asl_alphabet_train", ruta):
        if (candidata / "A").is_dir():
            return candidata
    raise FileNotFoundError(f"No encuentro las carpetas de clases bajo {ruta}")


def leer_split():
    """Lee datos/dataset_splits.csv (generado en la seccion 10 del EDA del Lab 3).

    Devuelve ({'train'|'val'|'test': (rutas_relativas, etiquetas int64)}, clases).
    """
    with open(SPLITS_CSV, newline="", encoding="utf-8") as f:
        filas = list(csv.DictReader(f))
    clases = sorted({r["class"] for r in filas})
    indice = {c: i for i, c in enumerate(clases)}
    splits = {}
    for s in ("train", "val", "test"):
        sel = [r for r in filas if r["split"] == s]
        splits[s] = ([r["rel_path"] for r in sel],
                     np.array([indice[r["class"]] for r in sel], dtype=np.int64))
    return splits, clases


def cargar_imagen(ruta):
    """Unidad de trabajo de la fase 1: un JPG -> uint8 (TAM, TAM, 3).

    Lee del disco, decodifica el JPEG, pasa a RGB y reduce con BOX (promedio
    de area, el equivalente de INTER_AREA que pedia la metodologia). Funcion de
    nivel superior y sin estado: se puede repartir con un Pool tal cual.
    """
    with Image.open(ruta) as im:
        return np.asarray(im.convert("RGB").resize((TAM, TAM), Image.Resampling.BOX))


def cargar_secuencial(carpeta, rutas):
    """Fase 1 de la linea base: un for sobre las imagenes, en un solo nucleo.

    Es exactamente lo que paralelo_cpu.cargar_paralelo reparte entre nucleos;
    vive aqui (y no en secuencial.py) para que comparar ambas no importe torch.
    """
    return np.stack([cargar_imagen(Path(carpeta) / r) for r in rutas])


def calentar_cache(carpeta, rutas):
    """Lee los bytes de cada archivo sin decodificarlos (no se cronometra).

    Deja los ~270 MB del split en el cache de disco del SO, para que TODAS las
    corridas de TODAS las versiones partan igual y la fase 1 mida trabajo de
    CPU (decodificar y reducir), no la velocidad del disco en frio.
    """
    for r in rutas:
        (Path(carpeta) / r).read_bytes()


def registrar(version, dispositivo, hilos, epocas, corrida, fase, segundos, detalle=""):
    """Agrega una fila a resultados/tiempos.csv (crea el archivo si no existe)."""
    nuevo = not TIEMPOS_CSV.exists()
    TIEMPOS_CSV.parent.mkdir(exist_ok=True)
    with open(TIEMPOS_CSV, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if nuevo:
            w.writerow(COLUMNAS)
        w.writerow([datetime.now().isoformat(timespec="seconds"), version, dispositivo,
                    hilos, epocas, corrida, fase, f"{segundos:.3f}", detalle])
