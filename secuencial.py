"""Version SECUENCIAL (linea base) del clasificador ASL del Lab 3 de Data Science.

Todo corre en UN solo nucleo de CPU:
  Fase 1 - carga:          lee y reduce los 22,022 JPG del split, uno por uno.
  Fase 2 - entrenamiento:  AlexNetASL en CPU, PyTorch limitado a 1 hilo.
  Fase 3 - evaluacion:     accuracy y F1 macro sobre validacion y prueba.

Cada fase se cronometra por separado y se agrega a resultados/tiempos.csv;
resumen.py promedia esas filas y calcula el speedup de la version optimizada.

Uso:
  python secuencial.py                       # 3 corridas, 3 epocas, 1 hilo
  python secuencial.py --hilos 32            # entrenamiento CPU multihilo (referencia
                                             # honesta para la GPU; la carga sigue en 1 nucleo)
  python secuencial.py --corridas 1 --epocas 1 --limite 500   # prueba rapida, no registra
  python secuencial.py --nucleos-p           # el proceso anclado a los nucleos de rendimiento
                                             # (version "secuencial_nucleosP"; ver abajo)

Por que --nucleos-p: esta CPU (Core Ultra 9 275HX) tiene 8 nucleos P y 16 nucleos E, y un
nucleo E hace este trabajo ~1.5x mas lento. Sin anclar, Windows decide donde corre el unico
hilo de la linea base; en la sesion final lo dejo sobre todo en nucleos E (127 s por epoca
contra 89 s en un nucleo P). Anclado a los P, la linea base es "el mejor nucleo unico", y el
speedup contra ella es el conservador.
"""
import argparse
import ctypes
import sys
import time

import torch

import preparacion as P
import red as R


def anclar_a_nucleos_p():
    """Ancla el proceso a los nucleos de mayor EfficiencyClass (los P) y devuelve sus indices.

    Los detecta con GetLogicalProcessorInformationEx(RelationProcessorCore) de Windows, en vez de
    suponer su numeracion (en esta CPU son 0, 1, 10-13, 22 y 23, no los 8 primeros).
    """
    if sys.platform != "win32":
        raise SystemExit("--nucleos-p solo esta implementado para Windows")
    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    largo = ctypes.c_ulong(0)
    k32.GetLogicalProcessorInformationEx(0, None, ctypes.byref(largo))   # 0 = RelationProcessorCore
    buf = ctypes.create_string_buffer(largo.value)
    if not k32.GetLogicalProcessorInformationEx(0, buf, ctypes.byref(largo)):
        raise OSError(ctypes.get_last_error(), "GetLogicalProcessorInformationEx")
    nucleos, pos = [], 0
    while pos < largo.value:
        tam = int.from_bytes(buf.raw[pos + 4:pos + 8], "little")       # .Size
        clase = buf.raw[pos + 9]                                         # .Processor.EfficiencyClass
        mascara = int.from_bytes(buf.raw[pos + 32:pos + 40], "little")   # .GroupMask[0].Mask
        nucleos.append((clase, mascara))
        pos += tam
    mejor = max(c for c, _ in nucleos)
    mascara_p = 0
    for clase, m in nucleos:
        if clase == mejor:
            mascara_p |= m
    k32.GetCurrentProcess.restype = ctypes.c_void_p
    k32.SetProcessAffinityMask.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
    if not k32.SetProcessAffinityMask(k32.GetCurrentProcess(), mascara_p):
        raise OSError(ctypes.get_last_error(), "SetProcessAffinityMask")
    return [i for i in range(64) if mascara_p >> i & 1]


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--corridas", type=int, default=3)
    ap.add_argument("--epocas", type=int, default=3)
    ap.add_argument("--hilos", type=int, default=1,
                    help="hilos de PyTorch en CPU; 1 = secuencial de verdad")
    ap.add_argument("--datos", help="carpeta del dataset (por defecto, cache de kagglehub)")
    ap.add_argument("--limite", type=int,
                    help="solo las primeras N imagenes de cada split (para probar; "
                         "estas corridas NO se registran)")
    ap.add_argument("--nucleos-p", action="store_true",
                    help="anclar el proceso a los nucleos de rendimiento (P) de la CPU")
    args = ap.parse_args()

    if args.nucleos_p:
        print(f"proceso anclado a los nucleos P: {anclar_a_nucleos_p()}")

    # Ambos son globales del proceso y deben fijarse antes de cualquier operacion de torch.
    torch.set_num_threads(args.hilos)
    torch.set_num_interop_threads(1)
    version = "secuencial" if args.hilos == 1 else f"cpu_{args.hilos}hilos"
    if args.nucleos_p:
        version += "_nucleosP"

    carpeta = P.ruta_dataset(args.datos)
    splits, clases = P.leer_split()
    if args.limite:
        splits = {s: (r[:args.limite], e[:args.limite]) for s, (r, e) in splits.items()}
    n_imgs = sum(len(r) for r, _ in splits.values())
    print(f"{version}: {n_imgs} imagenes, {len(clases)} clases, {args.epocas} epocas, "
          f"torch {torch.__version__}, {torch.get_num_threads()} hilo(s)")

    t0 = time.perf_counter()
    P.calentar_cache(carpeta, [r for rutas, _ in splits.values() for r in rutas])
    R.calentar(len(clases), "cpu")
    print(f"calentamiento (no se registra): {time.perf_counter() - t0:.1f} s")

    def anotar(corrida, fase, segundos, detalle=""):
        print(f"  [{fase}] {segundos:.2f} s  {detalle}")
        if not args.limite:
            P.registrar(version, "cpu", args.hilos, args.epocas, corrida, fase, segundos, detalle)

    for corrida in range(1, args.corridas + 1):
        print(f"\nCorrida {corrida}/{args.corridas}")
        t_total = time.perf_counter()

        # ---- Fase 1: preparacion de datos (CPU, un nucleo) ----
        t0 = time.perf_counter()
        datos = {s: (torch.from_numpy(P.cargar_secuencial(carpeta, rutas)), torch.from_numpy(etq))
                 for s, (rutas, etq) in splits.items()}
        anotar(corrida, "carga", time.perf_counter() - t0, f"imagenes={n_imgs}")

        # ---- Fase 2: entrenamiento (CPU) ----
        torch.manual_seed(R.SEMILLA)
        modelo = R.AlexNetASL(len(clases))
        X, y = datos["train"]
        t0 = time.perf_counter()
        por_epoca = R.entrenar(modelo, X, y, "cpu", args.epocas)
        anotar(corrida, "entrenamiento", time.perf_counter() - t0,
               "por_epoca=" + ";".join(f"{t:.2f}" for t in por_epoca))

        # ---- Fase 3: evaluacion ----
        t0 = time.perf_counter()
        acc_val, f1_val = R.evaluar(modelo, *datos["val"], "cpu")
        acc_test, f1_test = R.evaluar(modelo, *datos["test"], "cpu")
        anotar(corrida, "evaluacion", time.perf_counter() - t0,
               f"acc_val={acc_val:.4f};f1_val={f1_val:.4f};"
               f"acc_test={acc_test:.4f};f1_test={f1_test:.4f}")

        anotar(corrida, "total", time.perf_counter() - t_total)


if __name__ == "__main__":
    main()
