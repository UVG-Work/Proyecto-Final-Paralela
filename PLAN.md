# Proyecto 3 — Mejorando algoritmos con CUDA: plan de trabajo

Clasificador del alfabeto ASL (Lab 3 de Data Science): una versión **secuencial** y una
**optimizada** en la que la preparación de imágenes corre en **varios núcleos de CPU** y la
matemática de la red neuronal corre en la **GPU con CUDA**. Medimos el speedup de cada fase.

Lo que se califica (enunciado, sección V):

| Puntos | Qué | Quién lo produce |
|---|---|---|
| 40 | Código: separación clara *preparación de datos en CPU* / *entrenamiento en CUDA* | A (base), B (CPU), C (GPU) |
| 30 | Evidencia: logs + capturas (Administrador de tareas, `nvidia-smi`) de cada fase | sesión final, los tres |
| 30 | Informe PDF: tabla de tiempos, speedup por fase, discusión técnica | B integra, todos escriben |

---

## 1. Decisiones ya tomadas (no cambiarlas sin avisar al grupo)

1. **El código original del modelo no existe en el repo del Lab 3.** El repo
   (`rodrigoajmac/CC_3084_Laboratorio_3`) solo tiene el EDA y `dataset_splits.csv`. Por eso
   `secuencial.py` **reconstruye** el pipeline siguiendo la metodología documentada en las
   secciones 9 y 10 de ese EDA (limpieza ya aplicada en el split, 64×64, AlexNet adaptada,
   F1 macro). **Si alguien tiene el notebook de modelado original, avise ya**: habría que
   alinear el modelo con ese o justificar en el informe por qué no.
2. **Datos:** el split del Lab 3 (`datos/dataset_splits.csv`): 22,022 imágenes, 29 clases,
   15,415 train / 3,303 val / 3,304 test. Las imágenes salen de la caché de `kagglehub`
   (`grassknoted/asl-alphabet`).
3. **Modelo e hiperparámetros:** `AlexNetASL` en `red.py` (AlexNet adaptada a 64×64, con
   BatchNorm y sin dropout; 1,372,861 parámetros), Adam con OneCycle (lr máx. 1e-3), lote 64,
   **3 épocas** para todas las mediciones, semilla 42. Llega a ~99.5 % en test. La primera
   versión (sin BatchNorm, con dropout, lr constante) se quedaba en 64 % y está documentada
   en `docs/paso1_2_secuencial.md` como decisión.
4. **Línea base = 1 núcleo de verdad:** carga de imágenes con un `for` y PyTorch con
   `torch.set_num_threads(1)`.
5. **Una sola forma de medir:** cada script agrega filas a `resultados/tiempos.csv`
   (`version, fase, segundos, ...`) y `resumen.py` saca promedio, desviación y speedup por
   fase. Antes de la corrida 1 hay un **calentamiento que no se registra** (lee los bytes de
   todos los archivos y da un paso de entrenamiento), así que todas las corridas cuentan.
6. **Todas las mediciones finales en la misma máquina y en la misma sesión:** la PC de C
   (**Intel Core Ultra 9 275HX: 8 núcleos P + 16 E, 24 hilos, sin Hyper-Threading**; 32 GB;
   RTX 5070 Laptop 8 GB, Blackwell `sm_120`, 36 SM; driver 596.49). Ver sección 5.
   - **Cambio (15/11, C):** antes decía "la laptop de A (i9-14900HX, 32 hilos)". Las
     mediciones preliminares de A (`resultados/preliminar/`) y las estimaciones del paso 3
     son de esa otra laptop; la línea base definitiva se mide en esta PC. Los barridos ya no
     llegan a 32 (aquí hay 24 hilos).

## 2. Estructura del repo

| Archivo | Fase | Dueño | Qué es |
|---|---|---|---|
| `preparacion.py` | 1 · CPU | A | lectura del split, `cargar_imagen()` (un JPG → uint8 64×64×3), `cargar_secuencial()` (el `for` de la línea base), registro de tiempos. **No importa torch** (ver trampa 1). |
| `red.py` | 2 · red neuronal | A | `AlexNetASL`, `entrenar()`, `evaluar()`, `calentar()`. Reciben el dispositivo como parámetro: misma matemática en CPU y GPU. |
| `secuencial.py` | ambas | A | **línea base**: todo en 1 núcleo. `--hilos N` da la referencia CPU multihilo. |
| `resumen.py` | — | A | tabla de promedios y speedup por fase. |
| `docs/paso1_2_secuencial.md` | — | A | pasos 1 y 2 del enunciado (análisis + medición secuencial). |
| `paralelo_cpu.py` | 1 · CPU | **B** | `cargar_paralelo()` con varios núcleos. |
| `optimizado.py` | 1 + 2 | **C** | carga con `cargar_paralelo()` + entrenamiento/evaluación en CUDA. `--kernels-propios` = versión `optimizado_numba`. |
| `kernels_cuda.py` | 2 · GPU | **C** | kernels propios en CUDA Python (numba-cuda): inferencia completa de la red con grid/bloques diseñados a mano. `python kernels_cuda.py` verifica y mide cada kernel. |
| `perfilar_gpu.py` | 2 · GPU | **C** | grid/bloques reales de cada kernel (cuDNN, cuBLAS, PyTorch y los propios) con Nsight Compute (`ncu`). |
| `docs/paso3_*.md`, `docs/paso4_5_*.md` | — | B, C | sus pasos del enunciado. |

**Regla de oro:** `preparacion.py` y `red.py` definen *qué* se calcula y los usan las dos
versiones. No se cambian para acelerar solo una de ellas (eso invalida el speedup). Si hay
que cambiar algo ahí, se avisa y se vuelve a medir la línea base.

## 3. Instalación (cada quien en su máquina)

El entorno virtual va **fuera de OneDrive** (la sincronización de miles de archivos ensucia
las mediciones):

```bash
python -m venv C:\Users\<usuario>\.venvs\asl-cuda
C:\Users\<usuario>\.venvs\asl-cuda\Scripts\python -m pip install -r requirements.txt
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

`requirements.txt` instala PyTorch con CUDA 12.8 (el mínimo para la RTX 5070, que es
Blackwell). Probado en la laptop de A y en la PC final: `torch 2.11.0+cu128`,
`cuda.is_available() == True`.

**En la PC final** el venv ya existe: `C:\Users\hugob\.venvs\asl-cuda` (Python 3.12.10).
Ojo: el `python` del PATH es el de MSYS2 (3.14), que no tiene PyTorch con CUDA; usar
siempre `C:\Users\hugob\.venvs\asl-cuda\Scripts\python` o activar el venv primero
(`C:\Users\hugob\.venvs\asl-cuda\Scripts\Activate.ps1`).

`requirements.txt` también fija numba-cuda y las bibliotecas CUDA 12.8 para los kernels de C,
y `numpy<2.4` (ver trampas 6 y 7).
La primera ejecución descarga el dataset (~1 GB) con kagglehub; luego usa la caché.

Prueba rápida (no registra tiempos):

```bash
python secuencial.py --corridas 1 --epocas 1 --limite 300
```

---

## 4. Reparto

### A — Pasos 1 y 2: análisis y línea base secuencial ✅ (hecho)

- `preparacion.py`, `red.py`, `secuencial.py`, `resumen.py`, `requirements.txt`.
- `docs/paso1_2_secuencial.md`: partes del programa, tabla de operaciones, tabla de
  operaciones repetitivas con su conteo, procedimiento de medición, resultados preliminares y
  factores que influyen en el rendimiento.
- Medición **preliminar** en `resultados/preliminar/` (tomada con la laptop ocupada; sirve
  como orden de magnitud, **no** para el informe).
- Pendiente de A: correr la línea base definitiva en la sesión final (sección 5).

### B — Paso 3 + fase 1 paralela en CPU + integración del informe

**Entregable de código: `paralelo_cpu.py`** con esta interfaz exacta (C la va a importar):

```python
PROCESOS_OPTIMO = ...  # el mejor valor de tu barrido; C lo usa por defecto

def cargar_paralelo(carpeta, rutas, procesos=PROCESOS_OPTIMO) -> np.ndarray:
    """Mismo resultado que preparacion.cargar_secuencial(carpeta, rutas):
    uint8 (N, 64, 64, 3), en el MISMO orden que `rutas`.
    Crea y cierra su propio Pool; el arranque del Pool cuenta como parte de la carga."""
```

- Repartir `preparacion.cargar_imagen` entre núcleos. Recomendado: `multiprocessing.Pool.map`.
  `map` conserva el orden; `imap_unordered` **no** (desordena imágenes y etiquetas).
- Probar `chunksize` chicos (16, 64) contra el de defecto de `map` (`len // (procesos*4)`,
  bloques de ~230 imágenes con 24 procesos): bloques chicos reparten mejor entre núcleos P y E,
  que avanzan a distinta velocidad.
- Comparar también `concurrent.futures.ThreadPoolExecutor`: Pillow suelta el GIL al
  decodificar y redimensionar, así que los hilos también escalan y se ahorran el arranque de
  procesos (aunque el análisis de cabeceras JPEG sí es Python con GIL, así que se espera que
  escalen peor). La comparación procesos vs hilos es buen material para la discusión.
- `python paralelo_cpu.py --procesos 1 2 4 8 12 16 20 24`: barrido que registra la fase
  `carga` en `tiempos.csv` con `version = carga_procesos_<p>` (o `carga_hilos_<p>`) y
  **`epocas = 3`** (no afecta la carga, pero `resumen.py` empareja contra la línea base por
  `epocas` y `fase`). Cada medición carga las **22,022 rutas en una sola llamada**, con
  calentamiento previo (`preparacion.calentar_cache`), igual que la línea base.
- Comprobar **una vez** que `np.array_equal(cargar_paralelo(...), preparacion.cargar_secuencial(...))`.
- Paso 3 (`docs/paso3_oportunidades.md`): a partir de las tablas de A, qué operaciones se
  paralelizan, dónde (CPU o GPU) y por qué. Argumento central: la carga es *vergonzosamente
  paralela* (22,022 imágenes independientes) pero es E/S + decodificación JPEG con
  ramificaciones, que la CPU hace bien y la GPU no; la red es álgebra densa (convoluciones =
  multiplicaciones de matrices) que es justo lo que la GPU hace bien.
- Gráficas para el informe a partir de `resultados/resumen.csv` (tiempo por fase por
  versión, speedup de carga vs número de procesos con la curva de Amdahl).
- Integrar el PDF final (sección 6).

**Definición de terminado:** el barrido corre, la igualdad bit a bit da `True`, y hay una
gráfica de speedup de la carga contra número de procesos.

**Barrido preliminar en la PC final (15/11, lo corrió C; sin registrar en `tiempos.csv`,
archivado en `resultados/preliminar/barrido_275hx/`):**

| procesos | 1 | 2 | 4 | 8 | 12 | 16 | 20 | 24 | hilos 8 | hilos 16 | hilos 24 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| carga (s), media de 2 | 19.9 | 11.3 | 7.85 | **7.68** | 8.43 | 9.18 | 10.0 | 10.4 | 21.5 | 22.5 | 24.4 |

- La igualdad bit a bit da `True` (procesos e hilos).
- **El óptimo aquí es 8, no 16.** `PROCESOS_OPTIMO` se cambió a 8, con la tabla en el comentario.
- **La carga deja de escalar en 4–8 procesos (speedup ~2.6×), y los hilos no escalan nada.**
  Diagnóstico de C con 1 núcleo: abrir y leer el archivo cuesta 0.72 ms por imagen,
  decodificar 0.34 ms, convertir a RGB 0.02 ms y reducir 0.08 ms. Más de la mitad del tiempo
  es abrir archivos, y eso no escala, aun con la caché del disco caliente. El IPC no influye
  (devolver o no la imagen tarda lo mismo). El arranque del Pool crece de 0.03 s con 4
  procesos a 0.94 s con 24. La PC tiene **Sophos Intercept X** activo, y que cada apertura de
  archivo cueste tanto es consistente con su escaneo en acceso. **B:** es material directo
  para la discusión (factores que limitan) y para corregir en el paso 3 la estimación de
  2.5–3.5 s para la carga.

### C — Pasos 4 y 5: entrenamiento en GPU con CUDA ✅ (código hecho; sesión final medida el 15/11, ver docs/paso4_5_gpu.md sección 4)

**Estado (15/11):**
- `optimizado.py` corre completo en la PC final. En la prueba de validación (sin registrar,
  con la PC ocupada) dio: carga ~7.9 s, transferencia ~0.1 s, entrenamiento ~4.7 s
  (~1.5 s por época), evaluación ~0.25 s y total ~13 s. **Accuracy de test 99.61–99.73 %**
  (secuencial: 99.61 %). Cumple la definición de terminado de abajo.
- Corrida 0 de calentamiento implementada: su primera época tarda ~7 s contra 1.5 s de las
  demás, lo que confirma que hay que descartarla.
- **Pregunta "CUDA Python" resuelta con código, sin esperar al catedrático:** `kernels_cuda.py`
  tiene 5 kernels propios en numba-cuda (normalizar, conv+BN+ReLU fusionados con memoria
  compartida y bloqueo en registros, maxpool, GEMM por tiles y argmax), con grid y bloques
  diseñados a mano. Reproducen la red con error relativo de 1.5e-6 y 100 % de predicciones
  iguales a PyTorch. Tardan 25 ms por lote de 512, contra 19.5 ms de cuDNN; la primera versión
  tardaba 355 ms y la mejora está documentada.
  `python optimizado.py --kernels-propios` registra la versión `optimizado_numba`, cuya
  evaluación usa esos kernels (99.64 % en test).
- `docs/paso4_5_gpu.md` está escrito: diseño, configuración CUDA, tabla de kernels medida y
  razones de la diferencia de accuracy, y la tabla del perfilador con grid y bloques reales.
  Faltan los números de la sesión final.
- **Perfilador (resuelto 15/11):** `perfilar_gpu.py` usa **Nsight Compute** (`ncu`), porque
  `torch.profiler` falla en esta PC: su CUPTI 12.8 da `CUPTI_ERROR_INVALID_DEVICE` con este
  driver. Ya están habilitados los contadores de la GPU (*NVIDIA Control Panel → Desktop →
  Enable Developer Settings → Developer → Manage GPU Performance Counters → "Allow access to
  the GPU performance counters to all users" → Apply*); sin eso `ncu` da `ERR_NVGPUCTRPERM`.
  Tarda ~9 min con 5 lotes y deja `resultados/perfil/kernels.csv`.

**Plan original:**

**Entregable de código: `optimizado.py`**, mismas fases y mismas filas en `tiempos.csv` que
`secuencial.py`, con `version = optimizado`:

| fase | qué mide |
|---|---|
| `carga` | **una** llamada a `cargar_paralelo()` de B con las 22,022 rutas de los tres conjuntos concatenadas, y luego se separa por tamaños (no tres llamadas: serían tres arranques de Pool). Mientras B no termine, usar `preparacion.cargar_secuencial` |
| `transferencia` | copiar X/y de train, val y test a la GPU (CPU→GPU), con `torch.cuda.synchronize()` antes de parar el cronómetro |
| `entrenamiento` | `red.entrenar(modelo, X_gpu, y_gpu, "cuda", 3)` — **la misma función** que la línea base |
| `evaluacion` | `red.evaluar(..., "cuda")` |
| `total` | todo lo anterior |

- **Ya verificado por A en la GPU:** `red.entrenar/evaluar/calentar` funcionan tal cual con
  X/y ya en la GPU. Medido con la laptop ocupada: **~1.7 s por época** (contra ~240 s en 1
  núcleo de CPU) y 99.5–99.7 % en test, estable entre corridas.
- **Calentamiento:** fijar `torch.backends.cudnn.benchmark = True` *antes* de calentar; luego
  `preparacion.calentar_cache(...)` y una **corrida 0 completa sin registrar** (en GPU dura
  segundos). `red.calentar()` solo prueba lotes de 64, y la corrida real también usa el lote
  final de 55 y los de evaluación (512, 231, 232): cada forma nueva dispara el autoajuste de
  cuDNN y la carga perezosa de kernels. Medido: aun con `red.calentar()`, la primera época
  de la corrida 1 tardó 5.7 s contra 1.7 s de las demás.
- **No tocar el formato de memoria:** `red.a_tensor` ya entrega NCHW contiguo en GPU (con
  *channels_last*, BatchNorm en CUDA es 3.3x más lento: 27.5 contra 8.4 ms por lote) y deja
  *channels_last* en CPU, donde es 15 % más rápido. Es un buen ejemplo para la discusión de
  factores que limitan a CUDA.
- Configuración CUDA a documentar en el informe (es parte de los 40 puntos), **impresa al
  inicio de cada ejecución** para que quede en el log: nombre de la GPU,
  `torch.version.cuda`, `torch.backends.cudnn.version()`, `cudnn.benchmark`,
  `torch.cuda.max_memory_allocated()` al final; datos en la GPU desde el inicio (uint8,
  ~270 MB, caben de sobra en 8 GB).
- Ojo con los Tensor Cores: por defecto PyTorch ya ejecuta las convoluciones FP32 en TF32
  (`torch.backends.cudnn.allow_tf32 = True`), así que la versión "FP32" ya los usa.
  Documentarlo. Opcional, como versión **aparte** (`optimizado_amp`): `torch.autocast("cuda",
  dtype=torch.bfloat16)`. Reportar su accuracy al lado: si baja, hay que decirlo.
- El accuracy de GPU no será idéntico al de CPU, por razones a explicar: TF32, algoritmos de
  cuDNN no deterministas y otro orden de suma en punto flotante. (La CPU con 1 hilo sí es
  determinista: las 3 corridas preliminares dieron exactamente 99.61 %.)
- Paso 4 (`docs/paso4_5_gpu.md`): cómo se reparte el trabajo en hilos, bloques y grids. Con
  PyTorch esos tamaños los elige cuDNN/cuBLAS, así que **sacarlos del perfilador en vez de
  inventarlos**: `torch.profiler.profile(activities=[ProfilerActivity.CUDA])` sobre unos
  lotes y `export_chrome_trace(...)`; cada kernel trae sus `grid` y `block`. **En Windows
  esto falla de entrada** con `CUPTI_ERROR_INSUFFICIENT_PRIVILEGES` (probado en la laptop de
  A): hay que habilitar *Panel de control de NVIDIA → Escritorio → Habilitar configuración
  de desarrollador → Desarrollador → Administrar contadores de rendimiento de GPU → Permitir
  el acceso a todos los usuarios*, o correr la terminal como administrador. Explicar con
  eso: convolución → GEMM implícito (M = lote·alto·ancho, N = canales de salida,
  K = canales de entrada·k·k) repartido en *tiles* por bloque; ReLU/pooling → un hilo por
  elemento.
- **Pregunta abierta para el catedrático (C la hace esta semana):** ¿"CUDA Python" acepta
  PyTorch sobre CUDA, o espera al menos un kernel propio? El paso 4 pide *diseñar* el reparto
  en hilos, bloques y grids, y con PyTorch puro eso lo decide cuDNN, así que **lo probable es
  que sí haga falta uno**. Plan B mientras llega la respuesta: un kernel con `numba-cuda`
  (por ejemplo la capa densa final en inferencia) con su grid y bloques justificados a mano.
  Verificar primero que `numba-cuda` corre en la RTX 5070 (sm_120); si no, `cupy.RawKernel`.

**Definición de terminado:** `optimizado.py` corre 3 corridas completas, su accuracy en test
queda a menos de 1 punto de la secuencial (no idéntica, por las razones de arriba) y
`resumen.py` muestra speedup por fase.

### Los tres — Paso 6 e informe

- **El entregable 1 dice "un archivo .py o .ipynb"** con el código secuencial y el optimizado.
  Nosotros tenemos varios módulos (a propósito: la separación CPU/GPU se ve en la estructura).
  C le pregunta al catedrático, junto con lo de "CUDA Python", si el repo basta. Si no, B arma
  al final `entrega.ipynb`: una sección por fase (carga secuencial / paralela, entrenamiento
  CPU / GPU) que llama a los módulos y muestra sus salidas.
- **B** arma el PDF y las gráficas. **A** escribe la sección de pasos 1–2 (ya está en
  `docs/`). **C** escribe pasos 4–5 y responde las tres preguntas de discusión; **B** la
  parte de CPU. Cada quien revisa la sección de otro.
- Speedup **por fase** y total. Reportar dos speedups de entrenamiento: contra 1 núcleo
  (`secuencial`) y contra el **mejor** CPU multihilo (`cpu_<n>hilos`, ver sección 5). El
  primero es el que pide el enunciado; el segundo es el honesto, y la diferencia entre ambos
  es material de discusión. `resumen.py` calcula el primero; el segundo es dividir dos medias
  de la misma tabla.
- Cada quien hace commits de su propio trabajo en el repo compartido (el enunciado pide
  documentar contribuciones así).

---

## 5. Sesión final de medición (los tres, en la laptop de A)

Todo en una sola sesión para que las condiciones sean las mismas.

**Antes de empezar**
1. Cerrar **todo**: Blender, Roblox Studio, navegadores, Discord, Spotify, Codex, overlay de
   NVIDIA. (En la medición preliminar estaban abiertos: CPU al 25–50 % y GPU al ~70 % sin
   que corriera nada nuestro.)
2. Pausar la sincronización de OneDrive.
3. Conectar a la corriente y poner el modo de energía en *Máximo rendimiento*.
   Todos los comandos con el Python del venv (sección 3), no con el `python` del PATH.
4. Verificar reposo: Administrador de tareas con CPU < 5 % y `nvidia-smi` con GPU-Util ~0 %.
5. Arrancar con `resultados/tiempos.csv` **vacío** (mover el anterior a otra carpeta):
   `resumen.py` promedia todas las filas del archivo, incluidas las de pruebas o corridas
   abortadas.

**Logs de evidencia: `evidencia.ps1` (C, 15/11).** Envuelve cada comando. Mientras corre deja
en `resultados/evidencia/<version>/`:
- `log.txt`: la salida del script.
- `cpu.csv`: `typeperf`, % de cada procesador lógico cada 1 s.
- `gpu.csv`: `nvidia-smi` cada 200 ms.
- `captura_*.jpg`: capturas de pantalla automáticas cada N s, más una al final.
- `resumen.txt`: CPU y GPU en reposo antes de empezar, modo de energía, uso medio y máximo de
  la GPU, y temperatura máxima.

Antes de cada ejecución, dejar **en primer plano** el Administrador de tareas en la vista que
corresponda (ver la columna "Captura").

```powershell
& "$PSHOME\powershell.exe" -ExecutionPolicy Bypass -File .\evidencia.ps1 -Version secuencial -Comando "secuencial.py"
& "$PSHOME\powershell.exe" -ExecutionPolicy Bypass -File .\evidencia.ps1 -Version gpu_30epocas `
    -Comando "optimizado.py --epocas 30 --corridas 1 --no-registrar" -CadaSegundos 5 -VentanaNvidiaSmi
```

Equivale a lo que antes se hacía a mano: `typeperf "\Processor(*)\% Processor Time" -si 1 ...` y
`nvidia-smi --query-gpu=... --format=csv -lms 100` en otra terminal, más `Tee-Object` del log.

**Orden** (con 5 min de pausa entre ejecuciones; esta laptop baja el reloj cuando se calienta
y el sesgo cae siempre sobre lo último que se mide):

| # | Comando | Duración aprox. | Captura de pantalla |
|---|---|---|---|
| 1 | `python secuencial.py` | 40–55 min | Adm. de tareas, vista de procesadores lógicos: **un solo núcleo** ocupado durante el entrenamiento |
| 2 | `python secuencial.py --hilos <n> --corridas 1 --epocas 1` para n = 8, 16, 24 | ~10 min | — (en un CPU con núcleos P y E, 24 hilos no suele ser lo más rápido) |
| 3 | `python secuencial.py --hilos <mejor n>` | ~10 min | todos los núcleos ocupados |
| 4 | `python paralelo_cpu.py --procesos 1 2 4 8 12 16 20 24 --hilos 8 16 24` | ~8 min | todos los núcleos ocupados durante la carga. Si el óptimo cambia, actualizar `PROCESOS_OPTIMO` **antes** del paso 5 |
| 5 | `python optimizado.py` | ~2 min | CPU repartida durante la carga |
| 6 | `python optimizado.py --kernels-propios` | ~2 min | — (versión `optimizado_numba`) |
| 7 | `python optimizado.py --epocas 30 --corridas 1 --no-registrar` | ~2 min | solo para evidencia: con 3 épocas la GPU termina en segundos y no da tiempo de capturar `nvidia-smi` ni la pestaña GPU del Adm. de tareas bajo carga |
| 8 | `python kernels_cuda.py` y `python perfilar_gpu.py` | ~10 min (ncu es lento) | — (tablas del paso 4; no tocan `tiempos.csv`) |
| 9 | `python resumen.py` | — | la tabla final |

**Chequeos de sanidad antes de aceptar los números**
- `desv` pequeña frente a `media` en cada fase. Si una corrida se sale, repetir.
- La fase `carga` de `secuencial` y de `cpu_<n>hilos` debe salir **igual**: es el mismo código
  en 1 núcleo. Si no, la máquina cambió de estado entre ejecuciones.
- El accuracy de test de `optimizado` debe quedar a menos de 1 punto del secuencial.

## 6. Esqueleto del informe PDF

1. Introducción y hardware/software (CPU, GPU, RAM, versiones de Python, torch, CUDA, driver).
2. Pasos 1–2: análisis y medición secuencial (A).
3. Paso 3: oportunidades de paralelización (B).
4. Pasos 4–5: diseño CUDA (hilos/bloques/grids), configuración, implementación (C).
5. Resultados: tabla comparativa de tiempos, speedup por fase, gráficas (B).
6. Evidencia: capturas y extractos de logs por fase (los tres).
7. Discusión técnica — las tres preguntas del enunciado:
   - ¿Por qué procesar los archivos en varios núcleos de CPU y dejar **solo** la matemática de
     la red en la GPU?
   - ¿Cómo cambió CUDA el tiempo de procesamiento?
   - ¿Qué factores justifican o limitan usar CUDA? (transferencias, tamaño del lote, Amdahl
     con la fase de carga, arranque del contexto, memoria de 8 GB, throttling térmico.)
8. Librerías y funciones usadas para la GPU (torch.cuda, cuDNN, cuBLAS, profiler, numba si aplica).
9. Conclusiones.

## 7. Trampas conocidas

1. **Windows + multiprocessing = spawn.** Cada proceso hijo vuelve a importar el script
   principal (`__mp_main__`) y el módulo de la función que ejecuta. Por eso
   `preparacion.py` no importa torch. Regla: **ningún script que cree un Pool importa
   `torch`, `red` ni `secuencial` en el nivel superior**; en `optimizado.py` esos imports van
   dentro de `main()`. Si no, cada hijo carga PyTorch + las DLL de CUDA (segundos dentro de la
   fase `carga` cronometrada y mucha memoria comprometida; con muchos procesos puede terminar
   en `WinError 1455`). Y sin `if __name__ == "__main__":` el Pool se re-lanza a sí mismo.
2. **Llamadas CUDA asíncronas.** Sin `torch.cuda.synchronize()` antes de parar el cronómetro
   se mide solo el tiempo de encolar, y el speedup sale absurdamente alto. `red.entrenar` ya
   sincroniza al final de cada época; la fase de transferencia la tiene que sincronizar C.
3. **`.item()` dentro del ciclo** sincroniza la GPU en cada lote. `red.entrenar` ya acumula la
   pérdida en un tensor; no agregar `print(perdida.item())` dentro del ciclo.
4. **Núcleos P y E.** El Core Ultra 9 275HX tiene 8 núcleos de rendimiento y 16 de
   eficiencia, sin Hyper-Threading. Un proceso de 1 hilo puede caer en un núcleo E y medir más
   lento, y con muchos procesos los núcleos E pesan en el promedio. Es un factor a mencionar,
   no a esconder.
5. **Cambiar `preparacion.py` o `red.py`** obliga a re-medir la línea base (40–55 min).
6. **numba-cuda y torch deben usar la misma versión de CUDA (12.8).** torch carga su propio
   `nvJitLink_120_0.dll` 12.8; con el NVVM 12.9 que pip instala por defecto, el enlace de los
   kernels falla con `ERROR 4 in nvvmAddNVVMContainerToProgram`. Ya está fijado en
   `requirements.txt`.
7. **`numpy<2.4`.** numba-cuda 0.30 usa `np.row_stack`, que numpy 2.4 eliminó.
8. **Antivirus (Sophos Intercept X en la PC final).** Abrir cada archivo cuesta más que
   decodificarlo, y eso limita el escalado de la carga (ver el barrido en la sección 4, B). No
   desactivarlo para medir: es la condición real de la máquina. Se documenta como factor.
