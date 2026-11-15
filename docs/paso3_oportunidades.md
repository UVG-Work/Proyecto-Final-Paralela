# Paso 3 — Identificación de oportunidades de paralelización

**Responsable:** B.  
**Código asociado:** `paralelo_cpu.py`, `graficas.py`.  
**Referencias cruzadas:** Tablas de operaciones y conteos del análisis secuencial en `docs/paso1_2_secuencial.md` (A).

---

## 1. Clasificación de operaciones y potencial de paralelismo

A partir del desglose de 14 operaciones identificado en el Paso 1 (sección 1.2 de `docs/paso1_2_secuencial.md`), se clasifican las tareas del pipeline según su tipo de paralelismo y el hardware idóneo para su ejecución:

| Op. | Nombre | Naturaleza computacional | Tipo de paralelismo | Destino óptimo | Justificación técnica |
|---|---|---|---|---|---|
| 1 | Lectura del split | E/S ligera de metadatos (CSV, 470 KB) | Secuencial / trivial | CPU (1 hilo) | Dura < 5 ms; el costo de sincronizar o crear hilos supera el beneficio. |
| **2–6** | **Carga, decodificación JPEG, RGB, resize BOX y apilado** | E/S de disco + descompresión con ramificación (Huffman) + álgebra espacial simple | **Paralelismo de datos masivo (*Embarrassingly parallel*)** | **CPU (Multiproceso / Multihilo)** | 22,022 imágenes 100 % independientes. Alto grado de bifurcación condicional en la decodificación; interacción directa con el sistema de archivos del SO y la caché de disco. |
| 7 | Barajado de índices | Generación pseudoaleatoria de permutación | Secuencial | CPU / GPU | O(N) para 15,415 enteros (~60 KB); insignificante. |
| 8 | Normalización y formateo de lote | Conversión uint8→float32, transposición NCHW, división /255 | Paralelo por elemento (SIMD/SIMT) | GPU (CUDA) | Operación elemental altamente paralelizable en memoria contigua en la GPU. |
| **9, 11** | **Convoluciones (Forward y Backward)** | Álgebra lineal densa (GEMM), 99 % de los FLOPs del pipeline | **Paralelismo masivo de datos e instrucciones (SIMT / Tensor Cores)** | **GPU (CUDA)** | ~29 TFLOP en 3 épocas. Acceso homogéneo, altísima intensidad aritmética, sin ramificaciones dinámicas. |
| 10 | Función de pérdida (Cross-Entropy) | Reducción paralela (Softmax + log + suma) | Reducción SIMT | GPU (CUDA) | Se ejecuta por lote en paralelo reduciendo 29 logits a un escalar. |
| 12 | Actualización Adam + OneCycle | Elementwise sobre 1.37 M parámetros (grad, momentos m y v) | Paralelo por elemento (SIMT) | GPU (CUDA) | Operación bound por ancho de banda de memoria; la GPU ofrece ~500 GB/s frente a ~60 GB/s de la CPU. |
| **13** | **Inferencia (Forward de evaluación)** | GEMM denso idéntico a Op. 9 sin almacenamiento de activaciones | **SIMT / Tensor Cores** | **GPU (CUDA)** | 6,607 imágenes procesadas en lotes de 512; paralelismo masivo. |
| 14 | Cálculo de métricas (Accuracy, F1) | Conteo de aciertos y matriz de confusión reducida | Secuencial / Reducción | CPU | Reducción de 3,304 enteros; costo < 2 ms. |

---

## 2. Justificación técnica: ¿Dónde acelerar cada fase y por qué?

El diseño del sistema divide la aceleración en dos frentes arquitectónicos según las características de cada problema:

```
+--------------------------------------------------------------------------------+
| PIPELINE DE PROCESAMIENTO                                                      |
+--------------------------------------------------------------------------------+
|                                                                                |
|  [ 22,022 archivos JPG ]                                                       |
|             │                                                                  |
|             ▼                                                                  |
|  +──────────────────────────────────────────────────────────────────────────+  |
|  | FASE 1: PREPARACIÓN DE DATOS EN CPU (paralelo_cpu.py)                    |  |
|  | • 24 núcleos / 32 hilos (Intel i9-14900HX: 8P + 16E)                     |  |
|  | • Multiprocessing (Pool.map con chunksize dinámico)                      |  |
|  | • Lectura I/O, descompresión Huffman, redimensionamiento BOX uint8       |  |
|  +──────────────────────────────────────────────────────────────────────────+  |
|             │ Arreglos compactos uint8 (270.6 MB total en RAM)                 |
|             ▼                                                                  |
|  +──────────────────────────────────────────────────────────────────────────+  |
|  | TRANSFERENCIA A VRAM (PCIe Gen 4 x8/x16: ~15 GB/s)                       |  |
|  | • Envío de tensores uint8 compactos y conversión a float32 en VRAM       |  |
|  +──────────────────────────────────────────────────────────────────────────+  |
|             │                                                                  |
|             ▼                                                                  |
|  +──────────────────────────────────────────────────────────────────────────+  |
|  | FASE 2: ENTRENAMIENTO Y EVALUACIÓN EN GPU (optimizado.py)                |  |
|  | • NVIDIA GeForce RTX 5070 Laptop (CUDA 12.8, Blackwell / Ada)           |  |
|  | • Convoluciones implícitas GEMM con cuDNN y TF32 Tensor Cores           |  |
|  | • 1,372,861 parámetros actualizados con Adam en paralelo masivo          |  |
|  +──────────────────────────────────────────────────────────────────────────+  |
|                                                                                |
+--------------------------------------------------------------------------------+
```

### 2.1 ¿Por qué procesar la carga y preparación de imágenes en la CPU?

1. **Naturaleza del algoritmo de descompresión JPEG:**
   - La decodificación JPEG utiliza codificación de entropía basada en **tablas de Huffman** y codificación de longitud de carrera (RLE). Este proceso lee un flujo de bits de longitud variable de forma estrictamente secuencial bloque por bloque ($8 \times 8$).
   - Los procesadores de GPU sufren una penalización de rendimiento severa ante algoritmos con alta **divergencia de bifurcaciones (*branch divergence*)** y accesos a memoria desalineados. Si un hilo de un *warp* de 32 toma una rama condicional distinta (por un código de longitud diferente), todos los hilos del *warp* se serializan.
   - En contraste, los núcleos de CPU modernos (arquitectura x86_64 con predictores de saltos avanzados como los núcleos Raptor Cove P-cores) manejan eficientemente la lógica de control irregular de libjpeg/Pillow.

2. **Interacción con el sistema operativo y el sistema de archivos:**
   - Abrir 22,022 descriptores de archivo requiere llamadas al sistema (*syscalls* al kernel de Windows: `NtOpenFile`, `NtReadFile`).
   - La CPU tiene acceso directo y de baja latencia a la **caché de páginas del sistema operativo en memoria RAM**. 
   - Si se intentara cargar imágenes crudas en la GPU, se debería enviar cada JPG comprimido individualmente por el bus PCIe o transferir 22,022 buffers fragmentados, incurriendo en una sobrecarga de latencia catastrófica.

3. **Carga *Vergonzosamente Paralela* (*Embarrassingly Parallel*):**
   - No existe ninguna dependencia temporal ni espacial entre la imagen $i$ y la imagen $j$.
   - La partición del dominio es directa: si disponemos de $P$ procesos trabajadores, cada proceso puede decodificar $22,022 / P$ imágenes de forma totalmente desacoplada, sin necesidad de primitivas de sincronización, cerrojos ni comunicación inter-procesos durante la decodificación.

---

### 2.2 ¿Por qué ejecutar el entrenamiento y la evaluación en la GPU con CUDA?

1. **Dominancia del Álgebra Lineal Densa:**
   - Como se calculó en el análisis del Paso 1, las 5 capas convolucionales concentran el **99 % de toda la aritmética** del programa: $\approx 1.46 \times 10^{13}$ operaciones multiplicación-acumulación (MAC), equivalentes a **29.2 TFLOP** a lo largo de las 3 épocas.
   - Cada convolución 2D sobre un lote de tamaño $B=64$, canales $C$, mapa de activación $H \times W$ y kernel $K \times K$ se reformula matemáticamente como una multiplicación de matrices gigantesca (**GEMM**):
     $$\text{Matriz } A \in \mathbb{R}^{(B \cdot H_{out} \cdot W_{out}) \times (C_{in} \cdot K \cdot K)}, \quad \text{Matriz } B \in \mathbb{R}^{(C_{in} \cdot K \cdot K) \times C_{out}}$$
   - Las GPUs están específicamente diseñadas para ejecutar GEMMs de forma masivamente paralela mediante miles de núcleos CUDA y unidades especializadas (**Tensor Cores**).

2. **Paralelismo de Grano Fino (SIMT):**
   - En una convolución, cada píxel de salida y cada canal se calcula de forma independiente aplicando el mismo producto punto con la ventana del kernel. Esto encaja perfectamente en el paradigma **SIMT (Single Instruction, Multiple Threads)**: una única instrucción controla cientos de núcleos de cálculo que operan simultáneamente sobre datos contiguos en memoria.

3. **Ancho de banda de memoria:**
   - La CPU DDR5 del sistema entrega un ancho de banda teórico de $\sim 60\text{–}70\text{ GB/s}$.
   - La GPU dedicada NVIDIA RTX 5070 cuenta con memoria GDDR6/GDDR7 con un ancho de banda superior a $\mathbf{400\text{–}500\text{ GB/s}}$.
   - En capas como BatchNorm, ReLU y Max-Pooling, la velocidad de cómputo está limitada por la velocidad a la que se transfieren tensores de la memoria a los registros (*memory-bound*). El ancho de banda masivo de la GPU permite procesar estas capas en microsegundos.

---

## 3. Diseño de la paralelización de Fase 1 en CPU (`paralelo_cpu.py`)

### 3.1 Multiprocesamiento vs Multihilo en Python

Una decisión arquitectónica crítica en Python bajo sistemas Windows es cómo eludir el **GIL (Global Interpreter Lock)**:

| Aspecto | `multiprocessing.Pool` (Procesos) | `ThreadPoolExecutor` (Hilos) |
|---|---|---|
| **Efecto del GIL** | **Inexistente.** Cada proceso tiene su propio intérprete Python, su propio espacio de direcciones y su propio GIL. | **Parcial.** El GIL se bloquea durante la ejecución de bytecode de Python. Pillow libera el GIL en C durante el resize `BOX` y conversión de píxeles, pero el parseo de cabeceras JPEG compite por el GIL. |
| **Sobrecarga de creación** | **Alta en Windows.** Utiliza el método `spawn`, lo que implica arrancar un nuevo proceso `python.exe` y reimportar módulos base. | **Muy baja.** Hilos ligeros nativos de Windows dentro del mismo proceso. |
| **Comunicación (IPC)** | Los resultados deben serializarse (pickle) y enviarse por sockets/tuberías locales al proceso padre. | Memoria compartida directa en el mismo espacio virtual; sin costo de IPC. |
| **Escalabilidad en 22,022 imágenes** | **Óptima.** La sobrecarga inicial de `spawn` ($\sim 0.3\text{ s}$) se amortiza completamente sobre las 22,022 imágenes. | **Limitada.** Se observa saturación y contención del GIL cuando el número de hilos supera los 8–12. |

**Conclusión experimental:** 
Para lotes pequeños (< 500 imágenes), los hilos resultan más veloces debido a la ausencia de tiempo de arranque. Sin embargo, para el dataset completo de 22,022 imágenes, `multiprocessing.Pool` es superior porque los 24 núcleos de CPU pueden decodificar simultáneamente a máxima potencia sin contención del intérprete.

### 3.2 Granularidad y Balanceo de Carga (`chunksize`) en Arquitecturas Híbridas (P-Cores + E-Cores)

El procesador de pruebas es un **Intel Core i9-14900HX**, el cual posee una arquitectura heterogénea:
- **8 P-cores (Performance Cores):** Alta frecuencia ($\sim 5.0\text{ GHz}$), soporte de 2 hilos por núcleo (Hyper-Threading).
- **16 E-cores (Efficiency Cores):** Frecuencia moderada ($\sim 3.8\text{ GHz}$), 1 hilo por núcleo, optimizados para throughput multihilo.

Si se utiliza el `chunksize` por defecto de `multiprocessing.Pool.map` ($\frac{N}{4 \cdot P} \approx 172$ imágenes por bloque para 32 trabajadores):
- Un bloque grande asignado a un núcleo E tardará más tiempo en completarse que el mismo bloque en un núcleo P.
- Los núcleos P terminarían temprano y quedarían ociosos esperando que los núcleos E finalicen sus últimos bloques (*straggler effect*).

**Estrategia implementada en `paralelo_cpu.py`:**
Se fija un `chunksize = 64` (o dinámico ajustado). Bloques más pequeños permiten que los núcleos P procesen más bloques en el mismo intervalo de tiempo, logrando un balanceo dinámico de carga natural entre núcleos heterogéneos sin incurrir en una sobrecarga excesiva de IPC.

---

## 4. Análisis de Límites Teóricos y Ley de Amdahl

La **Ley de Amdahl** establece el límite superior del speedup teórico alcanzable por un sistema paralelo en función de la fracción paralelizable ($P$) y la fracción estrictamente secuencial ($1 - P$):

$$S(N) = \frac{1}{(1 - P) + \frac{P}{N}}$$

Cuando $N \to \infty$, el speedup máximo está acotado asintóticamente por:
$$S_{max} = \frac{1}{1 - P}$$

### 4.1 Desglose de la Línea Base Secuencial (1 núcleo CPU)

De acuerdo con las mediciones preliminares de `docs/paso1_2_secuencial.md` (Media de 3 corridas a 3 épocas):

| Fase | Tiempo promedio ($T_{seq}$) | Porcentaje del tiempo total |
|---|---|---|
| **Carga de imágenes** | $35.41\text{ s}$ | $3.93\text{ \%}$ |
| **Entrenamiento (3 épocas)** | $825.27\text{ s}$ | $91.55\text{ \%}$ |
| **Evaluación (val + test)** | $40.71\text{ s}$ | $4.52\text{ \%}$ |
| **Total** | $\mathbf{901.43\text{ s}}$ | $\mathbf{100.00\text{ \%}}$ |

---

### 4.2 Escenario 1: Acelerar ÚNICAMENTE el entrenamiento con CUDA (sin paralelizar la carga)

Si asumimos idealmente que la GPU reduce el tiempo de entrenamiento a casi cero ($T_{gpu\_train} \approx 0\text{ s}$), las fases de carga ($35.41\text{ s}$) y evaluación secuencial ($40.71\text{ s}$) actuarían como el cuello de botella secuencial ($1 - P$):

$$T_{restante} \ge 35.41\text{ s} + 40.71\text{ s} = 76.12\text{ s}$$
$$S_{max\_solo\_gpu} = \frac{901.43\text{ s}}{76.12\text{ s}} \approx \mathbf{11.84\times}$$

> **Conclusión crítica:** Aunque la GPU calcule a velocidad infinita, el speedup global nunca podría superar $\approx 11.8\times$. La fase de carga en un solo núcleo (35.4 s) pasaría a representar casi la mitad del tiempo total.

---

### 4.3 Escenario 2: Acelerar AMBAS fases (Carga en CPU multinúcleo + Entrenamiento en GPU CUDA)

Al paralelizar la carga en la CPU (Rol B) y el entrenamiento y evaluación en la GPU (Rol C):
1. **Fase 1 (Carga multihilo/multiproceso):**
   - Con 16 procesos en la CPU i9-14900HX, la carga desciende de $35.41\text{ s}$ a $\approx 2.5\text{–}3.5\text{ s}$ (un speedup de fase de $\mathbf{10\times\text{–}14\times}$).
2. **Transferencia PCIe:**
   - La copia de los arreglos compactos a la GPU requiere $\approx 0.1\text{–}0.2\text{ s}$.
3. **Fase 2 (Entrenamiento en GPU con CUDA):**
   - Medido preliminarmente en la RTX 5070: $\approx 1.7\text{ s}$ por época $\times 3 = \mathbf{5.1\text{ s}}$ (frente a $825.3\text{ s}$ en CPU: speedup de entrenamiento $\approx \mathbf{160\times}$).
4. **Fase 3 (Evaluación en GPU):**
   - Los 6,607 ejemplos de validación y test se evalúan en $\approx 0.4\text{ s}$ (frente a $40.7\text{ s}$).

**Tiempo total estimado optimizado:**
$$T_{opt} \approx 3.0\text{ s (carga)} + 0.2\text{ s (transfer)} + 5.1\text{ s (train)} + 0.4\text{ s (eval)} \approx \mathbf{8.7\text{ s}}$$

**Speedup global del sistema completo:**
$$S_{global} = \frac{901.43\text{ s}}{8.7\text{ s}} \approx \mathbf{103.6\times}$$

La paralelización conjunta de la CPU y la GPU eleva el speedup de **$11.8\times$ a más de $100\times$**, demostrando de forma contundente la importancia de no descuidar la fase de preparación de datos en la CPU en pipelines de Deep Learning de alto rendimiento.

