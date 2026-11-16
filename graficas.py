"""Graficas del informe final a partir de la sesion de medicion (B; corregido por C el 15/11).

Lee resultados/resumen.csv (salida de resumen.py) y resultados/evidencia/ y genera en
resultados/graficas/:
  1. tiempo_por_fase.png    tiempo medio de cada fase por version (un panel por fase,
                            cada uno con su escala: la GPU es 85x mas rapida y en una sola
                            escala sus barras serian invisibles)
  2. speedup_por_fase.png   speedup de la version optimizada contra 1 nucleo y contra el
                            mejor CPU multihilo (16 hilos)
  3. speedup_carga_cpu.png  escalabilidad de la carga: procesos vs hilos, con el speedup
                            ideal y la curva de Amdahl ajustada a lo medido
  4. uso_gpu.png            % de uso de la GPU durante la ejecucion de optimizado.py
                            (nvidia-smi ~5 muestras/s, resultados/evidencia/gpu_traza/gpu.csv)

Correcciones respecto de la version anterior:
- Se filtra epocas == 3: cpu_16hilos aparece tambien en el barrido de 1 epoca y antes se
  tomaba esa fila. Las versiones carga_* (solo fase de carga) ya no entran a la grafica
  de fases.
- El speedup de la carga se calcula contra 1 proceso del mismo Pool (misma maquinaria,
  solo cambia N), y la curva de Amdahl usa la fraccion paralela AJUSTADA a los datos,
  no valores supuestos.

Uso:
  python graficas.py
  python graficas.py --csv otra_ruta/resumen.csv --salida otra_carpeta
"""
import argparse
import os
import tempfile
from pathlib import Path

if "MPLCONFIGDIR" not in os.environ:
    os.environ["MPLCONFIGDIR"] = tempfile.gettempdir()

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parent
DEFAULT_RESUMEN = RAIZ / "resultados" / "resumen.csv"
OUTPUT_DIR = RAIZ / "resultados" / "graficas"
# Traza de uso de la GPU: corrida de optimizado.py sin registrar, hecha despues de corregir
# evidencia.ps1 (la de la medicion oficial perdio los ultimos ~9 s del log de nvidia-smi).
GPU_CSV = RAIZ / "resultados" / "evidencia" / "gpu_traza" / "gpu.csv"

# Estilo de articulo (como un informe en LaTeX): escala de grises y la tipografia Computer
# Modern del informe (informe/fuentes/). Los nombres conservan el papel de cada color.
SUPERFICIE = "#ffffff"
TINTA = "#000000"
TINTA_2 = "#333333"
REJILLA = "#dddddd"
AZUL = "#1f1f1f"          # GPU: casi negro
NARANJA = "#a6a6a6"       # CPU: gris
AZUL_OSCURO = "#1f1f1f"   # tres referencias de speedup, de oscuro a claro
AZUL_MEDIO = "#6b6b6b"
AZUL_CLARO = "#bdbdbd"
GRIS_REF = "#7a7a7a"
FUENTES = RAIZ / "informe" / "fuentes"

VERSIONES = [  # (version en el CSV, etiqueta, dispositivo)
    ("secuencial", "Secuencial, 1 núcleo\n(lo elige Windows: E)", "CPU"),
    ("secuencial_nucleosP", "Secuencial, 1 núcleo\n(anclado a núcleos P)", "CPU"),
    ("cpu_16hilos", "CPU 16 hilos", "CPU"),
    ("optimizado", "Optimizado\n(GPU, cuDNN)", "GPU"),
    ("optimizado_numba", "Optimizado\n(GPU, kernels propios)", "GPU"),
]
FASES = [("carga", "Carga de imágenes (CPU)"), ("entrenamiento", "Entrenamiento (3 épocas)"),
         ("evaluacion", "Evaluación (val + test)"), ("total", "Total")]


def estilo():
    from matplotlib import font_manager
    familia = "serif"
    for f in ("cmunrm.ttf", "cmunbx.ttf", "cmunti.ttf"):
        if (FUENTES / f).exists():
            font_manager.fontManager.addfont(str(FUENTES / f))
            familia = "CMU Serif"
    plt.rcParams.update({
        "font.family": familia, "mathtext.fontset": "cm", "axes.unicode_minus": False,
        "figure.facecolor": SUPERFICIE, "axes.facecolor": SUPERFICIE,
        "savefig.facecolor": SUPERFICIE, "font.size": 10,
        "text.color": TINTA, "axes.labelcolor": TINTA_2,
        "xtick.color": TINTA_2, "ytick.color": TINTA_2,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.titleweight": "normal", "axes.titlesize": 11, "axes.edgecolor": "#000000",
    })


def fmt_s(v):
    return f"{v:.2f} s" if v < 10 else f"{v:.1f} s"


def media(df, version, fase):
    fila = df[(df.version == version) & (df.fase == fase)]
    return float(fila.media.iloc[0]) if not fila.empty else np.nan


def desv(df, version, fase):
    fila = df[(df.version == version) & (df.fase == fase)]
    return float(fila.desv.iloc[0]) if not fila.empty else 0.0


def graficar_tiempo_fases(df, salida):
    df = df[df.epocas == 3]
    versiones = [v for v in VERSIONES if (df.version == v[0]).any()]
    fig, ejes = plt.subplots(2, 2, figsize=(11, 8.2), dpi=200)
    etiquetas = [e for _, e, _ in versiones]
    colores = [AZUL if d == "GPU" else NARANJA for _, _, d in versiones]
    y = np.arange(len(versiones))[::-1]
    for ax, (fase, titulo) in zip(ejes.flat, FASES):
        vals = [media(df, v, fase) for v, _, _ in versiones]
        errs = [desv(df, v, fase) for v, _, _ in versiones]
        ax.barh(y, vals, height=0.62, color=colores, edgecolor=SUPERFICIE, linewidth=2,
                xerr=errs, error_kw={"ecolor": TINTA_2, "elinewidth": 1, "capsize": 2})
        tope = max(v + e for v, e in zip(vals, errs))
        for yi, v, e in zip(y, vals, errs):   # la etiqueta va despues de la barra de error
            ax.text(v + e + tope * 0.02, yi, fmt_s(v), va="center", fontsize=9, color=TINTA)
        ax.set_yticks(y, etiquetas, fontsize=8.5)
        ax.set_xlim(0, tope * 1.22)
        ax.set_title(titulo, loc="left")
        ax.set_xlabel("segundos (media de 3 corridas)", fontsize=8.5)
        ax.grid(axis="x", color=REJILLA, linewidth=0.8)
        ax.set_axisbelow(True)
        ax.tick_params(axis="y", length=0)
    from matplotlib.patches import Patch
    fig.legend(handles=[Patch(color=NARANJA, label="Matemática de la red en CPU"),
                        Patch(color=AZUL, label="Matemática de la red en GPU (CUDA)")],
               loc="upper right", ncol=2, frameon=False, fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    guardar(fig, salida / "tiempo_por_fase.png")


def graficar_speedup_fases(df, salida):
    df = df[df.epocas == 3]
    fases = [f for f, _ in FASES]
    nombres = ["Carga\n(CPU, 4 procesos)", "Entrenamiento\n(GPU)", "Evaluación\n(GPU)", "Total"]
    series = [("secuencial", AZUL_OSCURO, "contra 1 núcleo elegido por Windows (E)"),
              ("secuencial_nucleosP", AZUL_MEDIO, "contra 1 núcleo P (el mejor núcleo único)"),
              ("cpu_16hilos", AZUL_CLARO, "contra CPU con 16 hilos")]
    series = [s for s in series if (df.version == s[0]).any()]
    vs_1 = [media(df, "secuencial", f) / media(df, "optimizado", f) for f in fases]
    fig, ax = plt.subplots(figsize=(9.5, 5), dpi=200)
    x = np.arange(len(fases))
    w = 0.8 / len(series)
    for i, (base, color, etiqueta) in enumerate(series):
        vals = [media(df, base, f) / media(df, "optimizado", f) for f in fases]
        desplaz = (i - (len(series) - 1) / 2) * w
        ax.bar(x + desplaz, vals, width=w, color=color, edgecolor=SUPERFICIE, linewidth=2,
               label=etiqueta)
        for xi, v in zip(x + desplaz, vals):
            ax.text(xi, v + 1.2, f"{v:.1f}$\\times$", ha="center", va="bottom", fontsize=8.5, color=TINTA)
    ax.set_xticks(x, nombres)
    ax.set_ylabel("speedup (veces más rápido)")
    ax.set_ylim(0, max(vs_1) * 1.15)
    ax.grid(axis="y", color=REJILLA, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(axis="x", length=0)
    ax.legend(frameon=False, loc="upper right")
    fig.tight_layout()
    guardar(fig, salida / "speedup_por_fase.png")


def graficar_speedup_carga(df, salida):
    carga = df[df.fase == "carga"]

    def serie(prefijo):
        filas = carga[carga.version.str.startswith(prefijo)]
        n = filas.version.str.rsplit("_", n=1).str[-1].astype(int)
        return pd.Series(filas.media.values, index=n.values).sort_index()

    procesos = serie("carga_procesos_")
    hilos = serie("carga_hilos_")
    if procesos.empty:
        print("[!] No hay barrido de procesos en el CSV.")
        return
    base = procesos.loc[1]                       # 1 proceso del mismo Pool
    s_proc = base / procesos
    s_hilos = base / hilos
    # Fraccion paralela P de Amdahl ajustada por minimos cuadrados en la zona que
    # escala (N <= 4); despues la curva medida se aparta por el costo de arranque del Pool
    # y la apertura de archivos, que no escala.
    zona = s_proc[s_proc.index <= 4]
    candidatos = np.linspace(0.5, 0.999, 1000)
    error = [((1 / ((1 - p) + p / zona.index.values) - zona.values) ** 2).sum() for p in candidatos]
    p = candidatos[int(np.argmin(error))]

    fig, ax = plt.subplots(figsize=(9, 5), dpi=200)
    n = np.linspace(1, 24, 200)
    ax.plot(n, n, color=GRIS_REF, linestyle="--", linewidth=1.2, label="ideal (S = N)")
    ax.plot(n, 1 / ((1 - p) + p / n), color=GRIS_REF, linestyle=":", linewidth=1.6,
            label=f"Amdahl ajustada a N $\\leq$ 4 (P = {p:.0%})")
    ax.plot(s_proc.index, s_proc.values, "o-", color=AZUL, linewidth=2, markersize=7,
            markeredgecolor=SUPERFICIE, markeredgewidth=2, label="procesos (multiprocessing.Pool)")
    if not s_hilos.empty:
        ax.plot(s_hilos.index, s_hilos.values, "s-", color=NARANJA, linewidth=2, markersize=7,
                markeredgecolor=SUPERFICIE, markeredgewidth=2, label="hilos (ThreadPoolExecutor)")
    mejor = s_proc.idxmax()
    ax.annotate(f"{s_proc.max():.2f}$\\times$ con {mejor} procesos\n({procesos.loc[mejor]:.2f} s)",
                (mejor, s_proc.max()), xytext=(mejor + 3.5, s_proc.max() + 1.6), fontsize=9,
                color=TINTA, arrowprops={"arrowstyle": "-", "color": TINTA_2, "linewidth": 0.8})
    ax.set_xticks(sorted(set(s_proc.index) | set(s_hilos.index)))
    ax.set_xlim(0.5, 24.5)
    ax.set_ylim(0, 12)
    ax.set_xlabel("número de procesos / hilos")
    ax.set_ylabel(f"speedup contra 1 proceso ({base:.1f} s)")
    ax.grid(color=REJILLA, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    guardar(fig, salida / "speedup_carga_cpu.png")


def graficar_uso_gpu(salida, gpu_csv=GPU_CSV):
    if not gpu_csv.exists():
        print(f"[!] No existe {gpu_csv}; se omite uso_gpu.png")
        return
    d = pd.read_csv(gpu_csv, skipinitialspace=True)
    d.columns = ["hora", "uso", "uso_mem", "mem", "temp", "reloj", "potencia"]
    # nvidia-smi a veces devuelve "[Unknown Error]" en una muestra suelta: se descarta.
    uso = pd.to_numeric(d.uso.astype(str).str.replace("%", "").str.strip(), errors="coerce")
    t = pd.to_datetime(d.hora, format="%Y/%m/%d %H:%M:%S.%f", errors="coerce")
    validas = uso.notna() & t.notna()
    descartadas = int((~validas).sum())
    uso, t = uso[validas], t[validas]
    seg = (t - t.iloc[0]).dt.total_seconds()
    if descartadas:
        print(f"[i] uso_gpu: {descartadas} de {len(d)} muestras de nvidia-smi ilegibles, descartadas")
    fig, ax = plt.subplots(figsize=(10, 3.9), dpi=200)
    ax.fill_between(seg, uso, color=AZUL, alpha=0.15, linewidth=0)
    ax.plot(seg, uso, color=AZUL, linewidth=1.4)
    ax.set_ylim(0, 105)
    ax.set_xlim(0, seg.max())
    ax.set_ylabel("uso de la GPU (%)")
    ax.set_xlabel("segundos desde el inicio de optimizado.py\n"
                  "bloques cerca del 97 %: entrenamiento en GPU  ·  valles: carga de imágenes en CPU "
                  "(la GPU espera)", fontsize=9)
    ax.grid(color=REJILLA, linewidth=0.8)
    ax.set_axisbelow(True)
    fig.tight_layout()
    guardar(fig, salida / "uso_gpu.png")


def guardar(fig, ruta):
    ruta.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(ruta)
    plt.close(fig)
    print(f"[ok] {ruta}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--csv", type=Path, default=DEFAULT_RESUMEN, help="resumen.csv de resumen.py")
    ap.add_argument("--salida", type=Path, default=OUTPUT_DIR, help="carpeta de las imagenes")
    args = ap.parse_args()
    if not args.csv.exists():
        raise SystemExit(f"No existe {args.csv}: correr primero python resumen.py")
    df = pd.read_csv(args.csv)
    estilo()
    graficar_tiempo_fases(df, args.salida)
    graficar_speedup_fases(df, args.salida)
    graficar_speedup_carga(df, args.salida)
    graficar_uso_gpu(args.salida)


if __name__ == "__main__":
    main()
