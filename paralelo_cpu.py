"""Fase 1 paralela en CPU: preparacion y decodificacion de imagenes multihilo / multiproceso.

Este modulo implementa la carga optimizada en CPU requerida por el Rol B.
Distribuye la lectura y decodificacion de las 22,022 imagenes entre los nucleos
de CPU disponibles.

REGLAS DE ARQUITECTURA (Windows + multiprocessing):
1. NO importa 'torch', 'red' ni 'secuencial' en el nivel superior.
   En Windows (spawn), los procesos hijos reimportarian librerias pesadas
   y DLLs de CUDA, degradando el rendimiento o provocando fallos de memoria.
2. La funcion cargar_paralelo conserva el orden exacto de 'rutas' y entrega un
   arreglo uint8 (N, 64, 64, 3) identico bit a bit a preparacion.cargar_secuencial.
3. El arranque y cierre del Pool se cronometran dentro de cargar_paralelo.

Uso:
  python paralelo_cpu.py --procesos 1 2 4 8 16 24 32    # Barrido de procesos para tiempos.csv
  python paralelo_cpu.py --hilos 1 2 4 8 16 24 32       # Barrido comparativo de hilos (ThreadPoolExecutor)
  python paralelo_cpu.py --verificar                    # Comprueba igualdad bit a bit con secuencial
  python paralelo_cpu.py --corridas 1 --limite 500       # Prueba rapida sin registrar
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Pool
import os
from pathlib import Path
import time

import numpy as np

import preparacion as P

# Valor optimo determinado por barrido experimental (16 nucleos / hilos es el
# balance optimo en procesadores hibridos P+E antes de saturar memoria/IPC).
# C importa este valor por defecto.
PROCESOS_OPTIMO = 16


def cargar_paralelo(carpeta, rutas, procesos=PROCESOS_OPTIMO, chunksize=64) -> np.ndarray:
    """Mismo resultado que preparacion.cargar_secuencial(carpeta, rutas):
    uint8 (N, 64, 64, 3), en el MISMO orden que `rutas`.

    Crea y cierra su propio Pool; el arranque del Pool cuenta como parte de la carga.
    Usa multiprocessing.Pool.map que garantiza la conservacion estricta del orden.
    """
    carpeta_p = Path(carpeta)
    rutas_completas = [str(carpeta_p / r) for r in rutas]
    n_items = len(rutas_completas)

    if n_items == 0:
        return np.empty((0, P.TAM, P.TAM, 3), dtype=np.uint8)

    # Si chunksize es None o 0, usa bloques chicos para equilibrar nucleos P y E
    cs = chunksize if (chunksize and chunksize > 0) else max(1, n_items // (procesos * 4))

    with Pool(processes=procesos) as pool:
        resultado = pool.map(P.cargar_imagen, rutas_completas, chunksize=cs)

    return np.stack(resultado)


def cargar_paralelo_hilos(carpeta, rutas, hilos=PROCESOS_OPTIMO) -> np.ndarray:
    """Variante con concurrent.futures.ThreadPoolExecutor:
    Pillow libera el GIL durante la decodificacion de pixeles y redimensionamiento BOX,
    lo que permite a los hilos escalar sin el costo de serializacion IPC ni spawn de procesos.
    Sin embargo, el parsing inicial de cabeceras JPEG compite por el GIL.
    """
    carpeta_p = Path(carpeta)
    rutas_completas = [str(carpeta_p / r) for r in rutas]
    n_items = len(rutas_completas)

    if n_items == 0:
        return np.empty((0, P.TAM, P.TAM, 3), dtype=np.uint8)

    with ThreadPoolExecutor(max_workers=hilos) as executor:
        resultado = list(executor.map(P.cargar_imagen, rutas_completas))

    return np.stack(resultado)


def verificar_igualdad(carpeta, rutas_muestra):
    """Comprueba bit a bit que cargar_paralelo produzca exactamente lo mismo que cargar_secuencial."""
    print("Verificando igualdad bit a bit contra preparacion.cargar_secuencial...")
    t0 = time.perf_counter()
    arr_sec = P.cargar_secuencial(carpeta, rutas_muestra)
    t_sec = time.perf_counter() - t0

    t0 = time.perf_counter()
    arr_par = cargar_paralelo(carpeta, rutas_muestra, procesos=4, chunksize=16)
    t_par = time.perf_counter() - t0

    t0 = time.perf_counter()
    arr_hilos = cargar_paralelo_hilos(carpeta, rutas_muestra, hilos=4)
    t_hilos = time.perf_counter() - t0

    igual_par = np.array_equal(arr_sec, arr_par)
    igual_hilos = np.array_equal(arr_sec, arr_hilos)

    print(f"  Secuencial:  {t_sec:.3f} s (shape: {arr_sec.shape}, dtype: {arr_sec.dtype})")
    print(f"  Procesos(4): {t_par:.3f} s -> Igual: {igual_par}")
    print(f"  Hilos(4):    {t_hilos:.3f} s -> Igual: {igual_hilos}")

    if not (igual_par and igual_hilos):
        raise ValueError("ERROR: La carga paralela no coincide bit a bit con la secuencial.")
    print("  [OK] Verificacion bit a bit exitosa (ambas coinciden al 100%).\n")


def ejecutar_barrido(carpeta, todas_las_rutas, trabajadores, metodo="procesos",
                     chunksize=64, corridas=3, epocas=3, limite=None):
    """Ejecuta un barrido de carga variando el numero de procesos o hilos y registra en tiempos.csv."""
    n_imgs = len(todas_las_rutas)
    etiqueta = "procesos" if metodo == "procesos" else "hilos"

    for w in trabajadores:
        version = f"carga_{etiqueta}_{w}"
        print(f"\n--- Evaluando {version} ({w} {etiqueta}) sobre {n_imgs} imagenes ---")

        for c in range(1, corridas + 1):
            t0 = time.perf_counter()
            if metodo == "procesos":
                arr = cargar_paralelo(carpeta, todas_las_rutas, procesos=w, chunksize=chunksize)
            else:
                arr = cargar_paralelo_hilos(carpeta, todas_las_rutas, hilos=w)
            segundos = time.perf_counter() - t0

            detalle = f"imagenes={n_imgs};metodo={metodo}"
            if metodo == "procesos":
                detalle += f";chunksize={chunksize}"

            print(f"  [Corrida {c}/{corridas}] carga: {segundos:.3f} s ({len(arr)} imagenes)")

            if not limite:
                P.registrar(
                    version=version,
                    dispositivo="cpu",
                    hilos=w,
                    epocas=epocas,
                    corrida=c,
                    fase="carga",
                    segundos=segundos,
                    detalle=detalle,
                )


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--procesos", type=int, nargs="+",
                        help="Lista de cantidad de procesos para el barrido (ej: --procesos 1 2 4 8 16 24 32)")
    parser.add_argument("--hilos", type=int, nargs="+",
                        help="Lista de cantidad de hilos para barrido comparativo (ej: --hilos 1 2 4 8 16 24 32)")
    parser.add_argument("--corridas", type=int, default=3,
                        help="Numero de corridas por configuracion (default: 3)")
    parser.add_argument("--epocas", type=int, default=3,
                        help="Numero de epocas a anotar en tiempos.csv (default: 3, para match con linea base)")
    parser.add_argument("--chunksize", type=int, default=64,
                        help="Tamano de bloque para pool.map (default: 64)")
    parser.add_argument("--limite", type=int, default=None,
                        help="Limitar a N imagenes totales para prueba rapida (NO registra en tiempos.csv)")
    parser.add_argument("--datos", default=None,
                        help="Ruta explicita a la carpeta del dataset")
    parser.add_argument("--verificar", action="store_true",
                        help="Ejecuta la prueba de verificacion bit a bit contra secuencial")

    args = parser.parse_args()

    carpeta = P.ruta_dataset(args.datos)
    splits, _ = P.leer_split()

    # Concatenamos todas las rutas en el orden establecido: train, val, test (22,022 imagenes)
    todas_las_rutas = splits["train"][0] + splits["val"][0] + splits["test"][0]

    if args.limite:
        todas_las_rutas = todas_las_rutas[:args.limite]
        print(f"Modo prueba: limitado a {len(todas_las_rutas)} imagenes (NO se registrara en tiempos.csv)")

    if args.verificar:
        muestra = todas_las_rutas[:min(200, len(todas_las_rutas))]
        verificar_igualdad(carpeta, muestra)
        if not args.procesos and not args.hilos:
            return

    # Si no se paso ningun argumento de barrido, corremos una demostracion por defecto
    if not args.procesos and not args.hilos:
        args.procesos = [1, 2, 4, 8, min(16, os.cpu_count() or 16)]

    # Calentamiento de cache previo a cualquier medicion (igual que en secuencial.py)
    print("Calentando cache de disco...")
    t0 = time.perf_counter()
    P.calentar_cache(carpeta, todas_las_rutas)
    print(f"Cache caliente ({time.perf_counter() - t0:.2f} s). Iniciando mediciones.\n")

    if args.procesos:
        ejecutar_barrido(
            carpeta=carpeta,
            todas_las_rutas=todas_las_rutas,
            trabajadores=args.procesos,
            metodo="procesos",
            chunksize=args.chunksize,
            corridas=args.corridas,
            epocas=args.epocas,
            limite=args.limite,
        )

    if args.hilos:
        ejecutar_barrido(
            carpeta=carpeta,
            todas_las_rutas=todas_las_rutas,
            trabajadores=args.hilos,
            metodo="hilos",
            chunksize=args.chunksize,
            corridas=args.corridas,
            epocas=args.epocas,
            limite=args.limite,
        )


if __name__ == "__main__":
    main()

