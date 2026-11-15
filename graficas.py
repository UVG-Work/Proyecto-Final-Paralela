"""Generacion automatica de graficas tecnicas para el informe final (Rol B).

Lee 'resultados/resumen.csv' (o 'resultados/tiempos.csv') y genera:
1. 'speedup_carga_cpu.png': Speedup de la carga vs numero de nucleos/hilos
   comparado con la curva teorica de Amdahl y el speedup ideal lineal.
2. 'tiempo_por_fase.png': Comparacion del tiempo por fase entre versiones.
3. 'procesos_vs_hilos.png': Comparativa de escalabilidad entre multiprocessing y ThreadPoolExecutor.

Uso:
  python graficas.py                     # Lee resultados/resumen.csv y guarda en resultados/graficas/
  python graficas.py --csv otra_ruta.csv # Especifica archivo de resumen alternativo
"""
import argparse
import os
from pathlib import Path
import tempfile

# Configurar directorio temporal de matplotlib para evitar advertencias de permisos en Windows
if "MPLCONFIGDIR" not in os.environ:
    os.environ["MPLCONFIGDIR"] = tempfile.gettempdir()

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parent
DEFAULT_RESUMEN = RAIZ / "resultados" / "resumen.csv"
OUTPUT_DIR = RAIZ / "resultados" / "graficas"


def amadahl_speedup(p, n):
    """Calcula el speedup teorico segun la Ley de Amdahl:
    S(n) = 1 / ((1 - p) + (p / n))
    donde p es la fraccion paralelizable del codigo.
    """
    return 1.0 / ((1.0 - p) + (p / np.array(n, dtype=float)))


def graficar_speedup_carga(df, carpeta_salida):
    """Grafica el speedup de la fase de carga contra numero de procesos e hilos,
    junto con la curva de la Ley de Amdahl y el speedup ideal.
    """
    # Filtrar fases de carga
    df_carga = df[df["fase"] == "carga"].copy()
    if df_carga.empty:
        print("[!] No hay datos de fase 'carga' para graficar speedup.")
        return

    # Extraer numero de procesos o hilos desde el nombre de la version o columna hilos
    # formatos esperados: carga_procesos_N, carga_hilos_N
    procesos_data = []
    hilos_data = []

    for _, row in df_carga.iterrows():
        ver = str(row["version"])
        speedup = row.get("speedup", 1.0)
        media = row.get("media", 0.0)

        if "procesos" in ver:
            try:
                num = int(ver.split("_")[-1])
                procesos_data.append((num, speedup, media))
            except ValueError:
                pass
        elif "hilos" in ver and ver != "secuencial":
            try:
                num = int(ver.split("_")[-1])
                hilos_data.append((num, speedup, media))
            except ValueError:
                pass

    if not procesos_data and not hilos_data:
        print("[!] No se encontraron variantes de carga_procesos o carga_hilos en el CSV.")
        return

    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)

    # Determinar rango de cores
    todos_n = [x[0] for x in procesos_data] + [x[0] for x in hilos_data]
    max_cores = max(todos_n) if todos_n else 32
    x_ideal = np.linspace(1, max_cores, 100)

    # Curva ideal lineal
    ax.plot(x_ideal, x_ideal, "k--", alpha=0.5, label="Speedup ideal lineal (S=N)")

    # Curvas teoricas de Amdahl para p = 0.95 y p = 0.98
    ax.plot(x_ideal, amadahl_speedup(0.98, x_ideal), ":", color="tab:gray", label="Amdahl (P=98%)")
    ax.plot(x_ideal, amadahl_speedup(0.95, x_ideal), "-.", color="tab:brown", label="Amdahl (P=95%)")

    # Datos reales de Procesos
    if procesos_data:
        procesos_data.sort(key=lambda x: x[0])
        p_x = [x[0] for x in procesos_data]
        p_y = [x[1] for x in procesos_data]
        ax.plot(p_x, p_y, "o-", color="tab:blue", linewidth=2.2, markersize=7, label="Multiprocessing (Pool.map)")

    # Datos reales de Hilos
    if hilos_data:
        hilos_data.sort(key=lambda x: x[0])
        h_x = [x[0] for x in hilos_data]
        h_y = [x[1] for x in hilos_data]
        ax.plot(h_x, h_y, "s-", color="tab:orange", linewidth=2.0, markersize=6, label="Multithreading (ThreadPool)")

    ax.set_title("Escalabilidad de la Carga de Datos en CPU (Fase 1)\nSpeedup vs Número de Trabajadores", fontsize=12, fontweight="bold")
    ax.set_xlabel("Número de Procesos / Hilos de CPU", fontsize=11)
    ax.set_ylabel("Speedup ($T_{secuencial} / T_{paralelo}$)", fontsize=11)
    ax.set_xticks(sorted(list(set(todos_n))))
    ax.legend(frameon=True, facecolor="white", framealpha=0.9)
    ax.grid(True, linestyle="--", alpha=0.6)

    carpeta_salida.mkdir(parents=True, exist_ok=True)
    out_file = carpeta_salida / "speedup_carga_cpu.png"
    plt.tight_layout()
    plt.savefig(out_file)
    plt.close()
    print(f"[✓] Grafica generada: {out_file}")


def graficar_tiempo_fases(df, carpeta_salida):
    """Grafica de barras comparativa del tiempo acumulado y por fase."""
    versiones_clave = ["secuencial", "optimizado"]
    versiones_presentes = [v for v in df["version"].unique() if v in versiones_clave or "hilos" in v]

    if not versiones_presentes:
        versiones_presentes = list(df["version"].unique()[:4])

    fases = ["carga", "transferencia", "entrenamiento", "evaluacion"]
    colores = {"carga": "#4C72B0", "transferencia": "#55A868", "entrenamiento": "#C44E52", "evaluacion": "#8172B2"}

    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, ax = plt.subplots(figsize=(9, 5), dpi=300)

    posiciones = np.arange(len(versiones_presentes))
    bottoms = np.zeros(len(versiones_presentes))

    for fase in fases:
        tiempos = []
        for ver in versiones_presentes:
            sub = df[(df["version"] == ver) & (df["fase"] == fase)]
            tiempos.append(sub["media"].values[0] if not sub.empty else 0.0)
        tiempos = np.array(tiempos)
        ax.bar(posiciones, tiempos, bottom=bottoms, label=fase.capitalize(), color=colores.get(fase, "gray"), alpha=0.85)
        bottoms += tiempos

    ax.set_xticks(posiciones)
    ax.set_xticklabels(versiones_presentes, rotation=15, ha="right", fontsize=10)
    ax.set_ylabel("Tiempo de Ejecución (segundos)", fontsize=11)
    ax.set_title("Comparación de Tiempo por Fase entre Versiones", fontsize=12, fontweight="bold")
    ax.legend(title="Fase", frameon=True)
    ax.grid(True, linestyle="--", alpha=0.5, axis="y")

    carpeta_salida.mkdir(parents=True, exist_ok=True)
    out_file = carpeta_salida / "tiempo_por_fase.png"
    plt.tight_layout()
    plt.savefig(out_file)
    plt.close()
    print(f"[✓] Grafica generada: {out_file}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--csv", type=Path, default=DEFAULT_RESUMEN,
                        help="Ruta al archivo resumen.csv o tiempos.csv")
    parser.add_argument("--salida", type=Path, default=OUTPUT_DIR,
                        help="Carpeta donde se guardaran las imagenes")
    args = parser.parse_args()

    if not args.csv.exists():
        # Intentar cargar preliminary si no existe
        alt = RAIZ / "resultados" / "preliminar" / "resumen.csv"
        if alt.exists():
            print(f"[i] No se encontro {args.csv}, usando preliminar: {alt}")
            args.csv = alt
        else:
            print(f"[!] No se encontro el archivo de datos: {args.csv}")
            return

    df = pd.read_csv(args.csv)

    print(f"Generando graficas desde {args.csv}...")
    graficar_speedup_carga(df, args.salida)
    graficar_tiempo_fases(df, args.salida)
    print("Listo.")


if __name__ == "__main__":
    main()

