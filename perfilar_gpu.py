"""Perfil de los kernels CUDA (paso 4): hilos, bloques y grids REALES de cada kernel.

En el entrenamiento con PyTorch, cuDNN/cuBLAS eligen como repartir cada
operacion en la GPU. En vez de suponer esos tamanos, este script los mide con
Nsight Compute (ncu), el perfilador oficial de NVIDIA: entrena unos lotes,
evalua, y corre la inferencia con nuestros kernels propios (kernels_cuda.py),
todo dentro de un rango NVTX "perfil"; ncu registra cada kernel lanzado ahi con
su grid, bloque, registros por hilo, memoria compartida, ocupacion y duracion.

Por que ncu y no torch.profiler: en esta PC kineto falla con
CUPTI_ERROR_INVALID_DEVICE (la CUPTI 12.8 que trae torch no reconoce la RTX 5070
Laptop con el driver 596.49) y la traza sale sin kernels de GPU.

Requisitos:
- Nsight Compute instalado (viene con el CUDA Toolkit); se busca en el PATH y en
  C:\\Program Files\\NVIDIA Corporation\\Nsight Compute *.
- Acceso a los contadores de la GPU (si no, ERR_NVGPUCTRPERM): NVIDIA Control
  Panel -> Desktop -> Enable Developer Settings -> Developer -> Manage GPU
  Performance Counters -> "Allow access to the GPU performance counters to all
  users" -> Apply. O correr la terminal como administrador.

Nota de medicion: ncu ejecuta los kernels de a uno, con la cache vaciada y el
reloj fijo en la frecuencia base, para que las metricas sean reproducibles. Las
duraciones sirven para comparar kernels entre si, no como tiempo real de la
fase (ese lo dan optimizado.py y resumen.py).

Salidas en resultados/perfil/:
  ncu_crudo.csv   una fila por lanzamiento (salida directa de ncu)
  kernels.csv     una fila por kernel + configuracion distinta, ordenado por tiempo

Uso:  python perfilar_gpu.py [--lotes 5]
"""
import argparse
import csv
import glob
import os
import re
import shutil
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

RAIZ = Path(__file__).resolve().parent
SALIDA = RAIZ / "resultados" / "perfil"

METRICAS = ",".join([
    "gpu__time_duration.sum",
    "launch__registers_per_thread",
    "launch__shared_mem_per_block",
    "launch__waves_per_multiprocessor",
    "sm__maximum_warps_per_active_cycle_pct",            # ocupacion teorica
    "sm__warps_active.avg.pct_of_peak_sustained_active",  # ocupacion alcanzada
])


def buscar_ncu():
    ncu = shutil.which("ncu") or shutil.which("ncu.bat")
    if ncu:
        return ncu
    candidatos = sorted(glob.glob(r"C:\Program Files\NVIDIA Corporation\Nsight Compute *\ncu.bat"))
    if candidatos:
        return candidatos[-1]
    sys.exit("No encuentro ncu (Nsight Compute). Instalar el CUDA Toolkit o Nsight Compute.")


def carga_de_trabajo(lotes):
    """Lo que se perfila. Corre DENTRO de ncu (modo --carga)."""
    import torch
    import kernels_cuda as KC
    import red as R

    torch.backends.cudnn.benchmark = True
    dev = "cuda"
    n = lotes * R.LOTE
    # Datos sinteticos con la forma real: el reparto en la GPU depende de las
    # formas, no de los valores.
    g = torch.Generator(device=dev).manual_seed(0)
    X = torch.randint(0, 256, (n, 64, 64, 3), dtype=torch.uint8, device=dev, generator=g)
    y = torch.randint(0, 29, (n,), device=dev, generator=g)
    torch.manual_seed(R.SEMILLA)
    modelo = R.AlexNetASL(29)
    motor_lote = 512
    Xe = torch.randint(0, 256, (motor_lote, 64, 64, 3), dtype=torch.uint8, device=dev, generator=g)

    # Calentamiento fuera del rango: autoajuste de cuDNN y compilacion de numba
    # (esos kernels de prueba no son los que usa el entrenamiento real).
    R.entrenar(modelo, X, y, dev, 1, log=lambda *_: None)
    R.evaluar(modelo, Xe, y[:0].new_zeros(motor_lote), dev)
    motor = KC.InferenciaNumba(modelo.eval())
    motor.predecir(Xe)
    torch.cuda.synchronize()

    torch.cuda.nvtx.range_push("perfil")
    R.entrenar(modelo, X, y, dev, 1, log=lambda *_: None)           # 'lotes' pasos de Adam
    R.evaluar(modelo, Xe, y[:0].new_zeros(motor_lote), dev)          # 1 lote de 512, cuDNN
    motor = KC.InferenciaNumba(modelo.eval())
    motor.predecir(Xe)                                               # 1 lote de 512, propios
    torch.cuda.synchronize()
    torch.cuda.nvtx.range_pop()


def categoria(nombre):
    n = nombre.lower()
    if re.search(r"\bk_\w+", nombre) or "kernels_cuda" in n:
        return "propio (numba)"
    if any(k in n for k in ("conv", "xmma", "implicit", "wgrad", "dgrad", "fprop", "cudnn")):
        return "convolucion (cuDNN)"
    if any(k in n for k in ("gemm", "cutlass", "cublas")):
        return "GEMM densa (cuBLAS/CUTLASS)"
    if "batch_norm" in n or "bn_" in n or "batchnorm" in n:
        return "BatchNorm"
    if "pool" in n:
        return "max-pool"
    if "multi_tensor" in n or "adam" in n:
        return "optimizador Adam"
    if "reduce" in n or "softmax" in n or "nll" in n or "argmax" in n:
        return "reduccion / perdida"
    if "elementwise" in n or "vectorized" in n or "copy" in n or "fill" in n:
        return "elemento a elemento"
    return "otro"


def nombre_corto(nombre):
    m = re.search(r"\b(k_\w+?)(?:\$|\b|_\d)", nombre)
    if m:
        return m.group(1)
    nombre = re.sub(r"<.*", "", nombre)             # quita los argumentos de plantilla
    return nombre.replace("void ", "")[:80]


def numero(v):
    try:
        return float(str(v).replace(",", ""))
    except ValueError:
        return 0.0


def analizar(crudo):
    with open(crudo, encoding="utf-8", errors="replace") as f:
        lineas = [l for l in f if l.startswith('"')]     # ncu antepone avisos ==PROF==
    filas = list(csv.DictReader(lineas))
    filas = [f for f in filas if f.get("Kernel Name") and f.get("ID", "").isdigit()]
    if not filas:
        sys.exit(f"ncu no registro kernels (revisar {crudo}).")

    grupos = defaultdict(lambda: {"llamadas": 0, "ns_total": 0.0, "ocup": []})
    for f in filas:
        clave = (f["Kernel Name"], f["Grid Size"], f["Block Size"])
        g = grupos[clave]
        g["llamadas"] += 1
        g["ns_total"] += numero(f["gpu__time_duration.sum"])
        g["ocup"].append(numero(f["sm__warps_active.avg.pct_of_peak_sustained_active"]))
        g["registros"] = int(numero(f["launch__registers_per_thread"]))
        g["mem_compartida_B"] = int(numero(f["launch__shared_mem_per_block"]))
        g["ocupacion_teorica_%"] = numero(f["sm__maximum_warps_per_active_cycle_pct"])
        g["olas_por_SM"] = numero(f["launch__waves_per_multiprocessor"])

    tabla = []
    total = sum(g["ns_total"] for g in grupos.values())
    for (nombre, grid, bloque), g in grupos.items():
        dims_g = [int(numero(d)) for d in grid.strip("()").split(",")]
        dims_b = [int(numero(d)) for d in bloque.strip("()").split(",")]
        bloques = dims_g[0] * dims_g[1] * dims_g[2]
        hilos_b = dims_b[0] * dims_b[1] * dims_b[2]
        tabla.append({
            "categoria": categoria(nombre), "kernel": nombre_corto(nombre),
            "grid": "x".join(map(str, dims_g)), "bloque": "x".join(map(str, dims_b)),
            "bloques": bloques, "hilos_por_bloque": hilos_b, "hilos_total": bloques * hilos_b,
            "registros_por_hilo": g["registros"], "mem_compartida_B": g["mem_compartida_B"],
            "ocupacion_teorica_%": round(g["ocupacion_teorica_%"], 1),
            "ocupacion_alcanzada_%": round(sum(g["ocup"]) / len(g["ocup"]), 1),
            "olas_por_SM": round(g["olas_por_SM"], 2),
            "llamadas": g["llamadas"], "us_total": round(g["ns_total"] / 1000, 1),
            "pct_tiempo": round(100 * g["ns_total"] / total, 2),
            "kernel_completo": nombre,
        })
    tabla.sort(key=lambda r: -r["us_total"])
    with open(SALIDA / "kernels.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(tabla[0]))
        w.writeheader()
        w.writerows(tabla)
    return filas, tabla, total


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--lotes", type=int, default=5,
                    help="lotes de entrenamiento perfilados (ncu es lento: ~0.1-0.5 s por kernel)")
    ap.add_argument("--carga", action="store_true", help=argparse.SUPPRESS)  # modo hijo
    args = ap.parse_args()

    if args.carga:
        carga_de_trabajo(args.lotes)
        return

    SALIDA.mkdir(parents=True, exist_ok=True)
    crudo = SALIDA / "ncu_crudo.csv"
    cmd = [buscar_ncu(), "--nvtx", "--nvtx-include", "perfil/", "--metrics", METRICAS,
           "--csv", "--page", "raw", "--log-file", str(crudo),
           sys.executable, os.path.abspath(__file__), "--carga", "--lotes", str(args.lotes)]
    print("Perfilando con Nsight Compute (tarda unos minutos)...")
    r = subprocess.run(cmd, capture_output=True, text=True, errors="replace")
    salida = r.stdout + r.stderr
    if "ERR_NVGPUCTRPERM" in salida:
        sys.exit("ERR_NVGPUCTRPERM: falta habilitar el acceso a los contadores de la GPU "
                 "(ver el encabezado de este archivo).")
    if r.returncode != 0:
        sys.exit(f"ncu termino con codigo {r.returncode}:\n{salida[-2000:]}")

    filas, tabla, total = analizar(crudo)

    por_cat = defaultdict(lambda: [0, 0.0])
    for t in tabla:
        por_cat[t["categoria"]][0] += t["llamadas"]
        por_cat[t["categoria"]][1] += t["us_total"]
    print(f"\n{len(filas)} lanzamientos de kernel, {len(tabla)} configuraciones distintas "
          f"({args.lotes} lotes de entrenamiento + 1 lote de evaluacion cuDNN + 1 con kernels propios)")
    print(f"\n{'categoria':30s} {'lanzam.':>8s} {'% tiempo':>9s}")
    for c, (n, us) in sorted(por_cat.items(), key=lambda kv: -kv[1][1]):
        print(f"{c:30s} {n:8d} {100 * us * 1000 / total:8.1f}%")

    print(f"\nLas 20 configuraciones con mas tiempo de GPU:")
    print(f"{'%':>5} {'n':>4} {'grid':>14} {'bloque':>10} {'reg':>4} {'smem':>6} "
          f"{'ocup%':>6}  kernel")
    for t in tabla[:20]:
        print(f"{t['pct_tiempo']:5.1f} {t['llamadas']:4d} {t['grid']:>14} {t['bloque']:>10} "
              f"{t['registros_por_hilo']:4d} {t['mem_compartida_B']:6d} "
              f"{t['ocupacion_alcanzada_%']:6.1f}  {t['kernel']}")

    print("\nKernels propios (grid/bloque medidos por ncu, comparar con kernels_cuda.py):")
    for t in tabla:
        if t["categoria"] == "propio (numba)":
            print(f"  {t['kernel']:20s} grid {t['grid']:>14} bloque {t['bloque']:>8} "
                  f"reg {t['registros_por_hilo']:3d} smem {t['mem_compartida_B']:6d} B "
                  f"ocupacion {t['ocupacion_alcanzada_%']:5.1f}%")
    print(f"\nTabla: {SALIDA / 'kernels.csv'}\nCrudo: {crudo}")


if __name__ == "__main__":
    main()
