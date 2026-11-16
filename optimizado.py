"""Version OPTIMIZADA del clasificador ASL: preparacion en varios nucleos de CPU,
matematica de la red neuronal en la GPU con CUDA.

Separacion del pipeline (lo que pide la seccion V del enunciado):
  Fase 1 - carga (CPU, multiproceso): paralelo_cpu.cargar_paralelo reparte la
           lectura y decodificacion de los 22,022 JPG entre nucleos. Una sola
           llamada con las rutas de train+val+test (un solo arranque de Pool).
  Transferencia (CPU -> GPU): X uint8 e y int64 de los tres conjuntos viajan
           UNA vez a la memoria de la GPU (~270 MB); despues ya no hay copias
           de datos por lote. Se cronometra aparte porque es el costo de usar CUDA.
  Fase 2 - entrenamiento (GPU): red.entrenar(..., "cuda"), la MISMA funcion que
           usa la linea base; solo cambia el dispositivo.
  Fase 3 - evaluacion (GPU): red.evaluar(..., "cuda").

Cada fase se agrega a resultados/tiempos.csv con version = "optimizado" y
resumen.py calcula el speedup por fase contra "secuencial".

Uso:
  python optimizado.py                            # 3 corridas, 3 epocas (medicion oficial)
  python optimizado.py --corridas 1 --limite 500  # prueba rapida, NO registra
  python optimizado.py --epocas 30 --corridas 1 --no-registrar
                                                  # carga larga para capturar nvidia-smi
  python optimizado.py --kernels-propios          # version "optimizado_numba": la fase de
                                                  # evaluacion usa los kernels CUDA Python
                                                  # propios (kernels_cuda.py) en vez de cuDNN
"""
# IMPORTANTE (Windows + multiprocessing = spawn): cada proceso del Pool de la
# fase 1 vuelve a importar ESTE archivo como __mp_main__. Por eso aqui arriba
# solo hay modulos livianos sin torch; torch y red se importan dentro de main().
# Si no, cada hijo cargaria PyTorch + las DLL de CUDA dentro de la fase de
# carga cronometrada (y con muchos procesos, WinError 1455).
import argparse
import time

import numpy as np

import paralelo_cpu as PC
import preparacion as P


def configurar_cuda(torch):
    """Fija y devuelve la configuracion CUDA usada (se imprime para el log)."""
    if not torch.cuda.is_available():
        raise SystemExit("CUDA no esta disponible: revisar driver y que torch sea la build +cu128.")
    # cuDNN prueba varios algoritmos de convolucion para cada forma de entrada
    # nueva y se queda con el mas rapido. Las formas aqui son fijas (lotes de
    # 64, el ultimo de 55, y los de evaluacion), asi que conviene.
    torch.backends.cudnn.benchmark = True
    # Valores por defecto de PyTorch, fijados explicitamente para documentarlos:
    # las convoluciones FP32 se ejecutan en TF32 sobre los Tensor Cores; las
    # multiplicaciones de matrices (capas densas) se quedan en FP32 "highest".
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("highest")

    prop = torch.cuda.get_device_properties(0)
    return {
        "gpu": prop.name,
        "capacidad": f"sm_{prop.major}{prop.minor}",
        "multiprocesadores (SM)": prop.multi_processor_count,
        "memoria GPU": f"{prop.total_memory / 2**30:.1f} GiB",
        "torch": torch.__version__,
        "torch.version.cuda": torch.version.cuda,
        "cuDNN": torch.backends.cudnn.version(),
        "cudnn.benchmark": torch.backends.cudnn.benchmark,
        "cudnn.allow_tf32 (convoluciones)": torch.backends.cudnn.allow_tf32,
        "matmul.allow_tf32 (densas)": torch.backends.cuda.matmul.allow_tf32,
        "cudnn.deterministic": torch.backends.cudnn.deterministic,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--corridas", type=int, default=3)
    ap.add_argument("--epocas", type=int, default=3)
    ap.add_argument("--procesos", type=int, default=PC.PROCESOS_OPTIMO,
                    help=f"procesos de la fase 1 (por defecto el optimo de B: {PC.PROCESOS_OPTIMO})")
    ap.add_argument("--datos", help="carpeta del dataset (por defecto, cache de kagglehub)")
    ap.add_argument("--limite", type=int,
                    help="solo las primeras N imagenes de cada split (prueba; NO se registra)")
    ap.add_argument("--no-registrar", action="store_true",
                    help="no escribir en tiempos.csv (corridas largas solo para capturas)")
    ap.add_argument("--kernels-propios", action="store_true",
                    help="evaluacion con los kernels de kernels_cuda.py (numba-cuda)")
    args = ap.parse_args()
    registrar = not (args.limite or args.no_registrar)

    import torch
    import red as R

    config = configurar_cuda(torch)
    dispositivo = "cuda"
    version = "optimizado_numba" if args.kernels_propios else "optimizado"
    if args.kernels_propios:
        import kernels_cuda as KC     # compila los kernels con numba en la corrida 0
        evaluar = lambda m, X, y, _disp: KC.evaluar(m, X, y)
    else:
        evaluar = R.evaluar

    carpeta = P.ruta_dataset(args.datos)
    splits, clases = P.leer_split()
    if args.limite:
        splits = {s: (r[:args.limite], e[:args.limite]) for s, (r, e) in splits.items()}
    nombres = list(splits)                                # train, val, test
    todas = [r for s in nombres for r in splits[s][0]]
    tamanos = [len(splits[s][0]) for s in nombres]
    n_imgs = len(todas)

    print(f"{version}: {n_imgs} imagenes, {len(clases)} clases, {args.epocas} epocas, "
          f"carga con {args.procesos} procesos")
    print("Configuracion CUDA:")
    for k, v in config.items():
        print(f"  {k:34s} {v}")
    if not registrar:
        print("(esta ejecucion NO se registra en tiempos.csv)")

    def anotar(corrida, fase, segundos, detalle=""):
        print(f"  [{fase}] {segundos:.3f} s  {detalle}")
        if registrar and corrida > 0:
            P.registrar(version, dispositivo, args.procesos, args.epocas, corrida, fase,
                        segundos, detalle)

    def corrida_completa(corrida, log):
        """Una ejecucion del pipeline completo, con cada fase cronometrada."""
        torch.cuda.reset_peak_memory_stats()
        t_total = time.perf_counter()

        # ---- Fase 1: preparacion de datos (CPU, varios procesos) ----
        t0 = time.perf_counter()
        X_todo = PC.cargar_paralelo(carpeta, todas, procesos=args.procesos)
        cortes = np.cumsum(tamanos)[:-1]
        X_cpu = dict(zip(nombres, np.split(X_todo, cortes)))
        anotar(corrida, "carga", time.perf_counter() - t0, f"imagenes={n_imgs}")

        # ---- Transferencia CPU -> GPU (una sola vez) ----
        # Viajan uint8 (4x menos bytes que float32); la conversion a float y la
        # division por 255 ocurren despues, en la GPU, dentro de red.a_tensor.
        t0 = time.perf_counter()
        datos = {s: (torch.from_numpy(X_cpu[s]).to(dispositivo),
                     torch.from_numpy(splits[s][1]).to(dispositivo)) for s in nombres}
        torch.cuda.synchronize()   # las copias son asincronas: esperar a que terminen
        mb = sum(x.nbytes + y.nbytes for x, y in datos.values()) / 2**20
        anotar(corrida, "transferencia", time.perf_counter() - t0, f"MiB={mb:.1f}")

        # ---- Fase 2: entrenamiento (GPU) ----
        torch.manual_seed(R.SEMILLA)          # mismos pesos iniciales que la linea base
        modelo = R.AlexNetASL(len(clases))
        X, y = datos["train"]
        t0 = time.perf_counter()
        por_epoca = R.entrenar(modelo, X, y, dispositivo, args.epocas, log=log)
        anotar(corrida, "entrenamiento", time.perf_counter() - t0,
               "por_epoca=" + ";".join(f"{t:.2f}" for t in por_epoca))

        # ---- Fase 3: evaluacion (GPU) ----
        t0 = time.perf_counter()
        acc_val, f1_val = evaluar(modelo, *datos["val"], dispositivo)
        acc_test, f1_test = evaluar(modelo, *datos["test"], dispositivo)
        anotar(corrida, "evaluacion", time.perf_counter() - t0,
               f"acc_val={acc_val:.4f};f1_val={f1_val:.4f};"
               f"acc_test={acc_test:.4f};f1_test={f1_test:.4f}")

        anotar(corrida, "total", time.perf_counter() - t_total,
               f"max_mem_gpu_MiB={torch.cuda.max_memory_allocated() / 2**20:.0f}")

    # ---- Calentamiento (no se registra) ----
    # 1) cache de disco del SO, igual que la linea base;
    # 2) una corrida 0 COMPLETA: crea el contexto CUDA, carga los kernels de
    #    cuDNN/cuBLAS y deja que cudnn.benchmark ajuste TODAS las formas que usa
    #    la corrida real (lotes de 64, el ultimo de 55 y los de evaluacion 512,
    #    231, 232). red.calentar() solo cubre lotes de 64: con el, la primera
    #    epoca de la corrida 1 seguia tardando 3x mas que las demas.
    t0 = time.perf_counter()
    P.calentar_cache(carpeta, todas)
    print("\nCorrida 0 (calentamiento, no se registra)")
    corrida_completa(0, log=lambda *_: None)
    print(f"calentamiento total: {time.perf_counter() - t0:.1f} s")

    for corrida in range(1, args.corridas + 1):
        print(f"\nCorrida {corrida}/{args.corridas}")
        corrida_completa(corrida, log=print)


if __name__ == "__main__":
    main()
