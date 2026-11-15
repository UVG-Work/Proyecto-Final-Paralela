"""Promedia resultados/tiempos.csv por version y fase, y calcula el speedup de
cada fase contra la version 'secuencial' con el mismo numero de epocas.

Todas las corridas cuentan: el calentamiento (cache de disco, contexto CUDA)
ya ocurre antes de la corrida 1 y no se registra.

Uso:  python resumen.py [ruta/tiempos.csv]
      -> imprime la tabla y escribe resumen.csv junto al CSV leido
"""
import sys
from pathlib import Path

import pandas as pd

ENTRADA = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent / "resultados" / "tiempos.csv"
ORDEN_FASES = ["carga", "transferencia", "entrenamiento", "evaluacion", "total"]

df = pd.read_csv(ENTRADA)
# dropna=False: una fila sin 'epocas' debe verse (con speedup vacio), no desaparecer.
t = (df.groupby(["version", "epocas", "fase"], dropna=False).segundos
       .agg(n="count", media="mean", desv="std", minimo="min", maximo="max").reset_index())

base = (t[t.version == "secuencial"][["epocas", "fase", "media"]]
        .rename(columns={"media": "media_secuencial"}))
t = t.merge(base, on=["epocas", "fase"], how="left")
t["speedup"] = t.media_secuencial / t.media
t["_f"] = t.fase.map(lambda f: ORDEN_FASES.index(f) if f in ORDEN_FASES else len(ORDEN_FASES))
t = t.sort_values(["epocas", "version", "_f"]).drop(columns=["_f", "media_secuencial"])

print(t.round(3).to_string(index=False))
t.round(4).to_csv(ENTRADA.with_name("resumen.csv"), index=False)
