# Pasos 4 y 5 — Diseño e implementación con CUDA

**Responsable:** C.
**Código:** `optimizado.py` (pipeline optimizado), `kernels_cuda.py` (kernels propios en CUDA
Python con numba-cuda), `perfilar_gpu.py` (grid/bloques reales de cuDNN/cuBLAS).
**Referencias:** operaciones y conteos en `docs/paso1_2_secuencial.md` (A); qué se paraleliza
y dónde en `docs/paso3_oportunidades.md` (B).

---

## 1. Qué corre dónde

| Fase | Dónde | Cómo | Código |
|---|---|---|---|
| 1. Carga (leer, decodificar JPEG, RGB, reducir a 64×64) | CPU, varios procesos | `multiprocessing.Pool.map`, **una** llamada para las 22,022 rutas | `paralelo_cpu.cargar_paralelo` (B) |
| Transferencia | CPU → GPU, **una vez** | `tensor.to("cuda")` de X (uint8) e y (int64) de train/val/test, ~270 MB | `optimizado.py` |
| 2. Entrenamiento | GPU | `red.entrenar(..., "cuda")`, la **misma** función que la línea base: cuDNN (convoluciones), cuBLAS (densas), kernels de PyTorch (ReLU, pooling, BN, Adam) | `red.py` (A) |
| 3. Evaluación | GPU | `red.evaluar(..., "cuda")` (cuDNN) **o** nuestro motor de inferencia con kernels propios | `red.py` / `kernels_cuda.py` |

Decisiones:

- **Los datos viajan a la GPU una sola vez y en uint8.** Se copian 4× menos bytes que en
  float32, y después no hay copias por lote: `red.entrenar` indexa los lotes ya dentro de la GPU
  y `red.a_tensor` convierte a float32 y divide entre 255 allá. Train, val y test (~270 MB)
  caben de sobra en los 8 GB.
- **`optimizado.py` no importa `torch` en el nivel superior.** En Windows los procesos del Pool
  arrancan con *spawn* y reimportan el script principal; si importara torch, cada hijo
  cargaría las DLL de CUDA dentro de la fase de carga cronometrada.
- **Calentamiento:** se hace `cudnn.benchmark = True`, se calienta la caché de disco y se ejecuta
  una corrida 0 completa sin registrar. Así, crear el contexto CUDA, cargar cuDNN y cuBLAS de
  forma perezosa, compilar los kernels de numba y autoajustar cuDNN para cada forma de lote
  (64, 55, 512, 231, 232) quedan fuera de las corridas medidas.
- **Sincronización:** las llamadas CUDA son asíncronas. La transferencia termina con
  `torch.cuda.synchronize()` y `red.entrenar` sincroniza al final de cada época, así que los
  cronómetros miden trabajo terminado y no solo encolado.

## 2. Configuración CUDA

Se imprime al inicio de cada ejecución de `optimizado.py` y queda en `resultados/log_optimizado.txt`.

| Parámetro | Valor | Por qué |
|---|---|---|
| GPU | NVIDIA GeForce RTX 5070 Laptop, Blackwell `sm_120`, 36 SM, 8 GB | — |
| PyTorch / CUDA / cuDNN | torch 2.11.0+cu128 / CUDA 12.8 / cuDNN 9.19.0 | `+cu128`: CUDA 12.8 es el mínimo para Blackwell |
| `cudnn.benchmark` | `True` | las formas son fijas: cuDNN prueba algoritmos una vez y reutiliza el más rápido |
| `cudnn.allow_tf32` | `True` (por defecto) | las convoluciones FP32 corren en **TF32 sobre Tensor Cores** (mantisa de 10 bits) |
| `matmul` precisión | `highest` (por defecto) | las capas densas quedan en FP32 completo |
| Formato de memoria | NCHW contiguo | con *channels_last*, BatchNorm en CUDA era 3.3× más lento (27.5 contra 8.4 ms por lote; medido por A) |
| Tamaño de lote | 64 (entrenamiento) / 512 (evaluación) | el mismo que la línea base; no se cambió para favorecer a la GPU |
| Memoria máxima usada | ~822 MiB por corrida (de 8 GB; prueba del 15/11) | los ~258 MiB de datos en uint8 + modelo, Adam y activaciones del lote |
| Bibliotecas propias | numba-cuda 0.30.4 con NVVM/nvJitLink 12.8 | misma versión de CUDA que torch (ver `requirements.txt`) |

## 3. Diseño del reparto en hilos, bloques y grids

### 3.1 Kernels propios (`kernels_cuda.py`, numba-cuda)

Para diseñar el reparto nosotros mismos, en vez de dejarlo a cuDNN, reimplementamos el
*forward* completo de la red entrenada (la inferencia de la fase de evaluación) con 5 kernels
escritos en CUDA Python:

| Kernel | Un hilo calcula | Bloque | Grid | Lanzamientos por lote |
|---|---|---|---|---|
| `normalizar` | 1 valor de salida (n, c, y, x): uint8 NHWC → float NCHW / 255 | 16×16 = 256 | (N·3, 4, 4) | 1 |
| `conv_bn_relu` | 1 píxel de salida (y, x) en **16 canales** (n, co..co+15) | 16×16 = 256 | (N·Co/16, ⌈H/16⌉, ⌈W/16⌉) | 5 |
| `maxpool2` | 1 píxel de salida (n, c, y, x) | 16×16 = 256 | (N·C, ⌈H/32⌉, ⌈W/32⌉) | 3 |
| `densa` | 1 elemento de Y = XWᵀ + b | 16×16 = 256 | (⌈O/16⌉, ⌈N/16⌉) | 2 |
| `argmax` | 1 imagen (recorre 29 logits) | 128 | ⌈N/128⌉ | 1 |

Con un lote de N = 512, la primera convolución (3→32, 64×64) lanza 512·(32/16)·4·4 =
**16,384 bloques** de 256 hilos, es decir, **4.2 M hilos**, cada uno con 16 salidas. La GPU
tiene **36 SM**, así que hay cientos de bloques por SM y el planificador siempre tiene warps
listos para ocultar la latencia de memoria.

Por qué se diseñó así:

1. **Un hilo por elemento de salida.** Cada salida de una convolución, un pooling o una ReLU
   es independiente de las demás (sin condiciones de carrera ni atómicos). Es el paralelismo
   de datos más fino posible y escala con el tamaño del lote.
2. **Bloques de 16×16 = 256 hilos (8 warps).** Es múltiplo de 32 (sin warps incompletos), deja
   varios bloques residentes por SM para alternar entre warps mientras otros esperan memoria,
   y su forma 2D calza con el plano de la imagen. Una imagen de 64×64 se cubre exacto con 4×4
   bloques.
3. **`threadIdx.x` recorre la coordenada x**, la contigua en memoria en NCHW. Los 32 hilos de un
   warp leen y escriben 32 floats consecutivos (128 bytes), en una sola transacción
   **coalescida**.
4. **(imagen, canal) va en `grid.x`.** `gridDim.y` y `gridDim.z` tienen un tope de 65,535, y
   512 imágenes × 128 canales = 65,536 planos ya lo superan. `gridDim.x` admite 2³¹−1.
5. **Memoria compartida en la convolución.** Cada bloque copia una vez el parche de entrada
   que necesita, (16+K−1)² valores por canal (halo incluido, con ceros fuera de la imagen), de
   **4 canales de entrada por etapa**, junto con los pesos de sus 16 filtros para esos
   canales. Hay `syncthreads()` después de cargar y antes de sobrescribir con la etapa
   siguiente.
6. **Bloqueo en registros (16 canales de salida por hilo).** Cada valor del parche se lee una
   vez de la memoria compartida y se multiplica por los pesos de 16 filtros, acumulando en 16
   registros. Así se reutiliza K²·16 veces (144 con K = 3). Todos los hilos del bloque leen el
   mismo peso al mismo tiempo (difusión desde la memoria compartida). K, 16 y 4 son constantes
   de compilación (hay un kernel por cada K), así que el compilador desenrolla los ciclos
   internos y el acumulador vive en registros.
   - *Iteración medida:* la primera versión (un hilo = 1 canal de salida, 1 canal de entrada
     por etapa) tardaba **355 ms** por lote de 512 en el forward completo. Con esta versión
     tarda **25 ms** (14× menos). Recargaba el mismo parche una vez por cada canal de salida y
     sincronizaba en cada canal de entrada: el cuello de botella era el acceso a memoria, no
     la aritmética.
7. **Fusión conv + BatchNorm + ReLU.** En inferencia, BN es una transformación afín por canal y
   se incorpora a los pesos (w' = w·γ/√(σ²+ε), b' = β − μ·γ/√(σ²+ε)). La ReLU se aplica antes de
   escribir. Son tres operaciones en una sola pasada por memoria global, en lugar de tres
   kernels que leen y escriben el tensor completo.
8. **GEMM por *tiles* en las capas densas.** Cada bloque de 16×16 hilos calcula un tile de 16×16
   de Y. Recorre la dimensión K en pasos de 16, cargando un tile de X y otro de W a memoria
   compartida, así que cada dato leído de la memoria global se usa 16 veces. El tile de W se
   guarda transpuesto y con una columna de relleno (16×17) para que la escritura no tenga
   conflictos de banco.
9. **Solo float32.** Las GPU de consumo ejecutan FP64 a 1/64 de velocidad, así que los
   literales van como `float32(...)` para que numba no promueva a float64.
10. **Interoperabilidad sin copias.** Los tensores de PyTorch pasan a numba a través de
    `__cuda_array_interface__`, y los kernels se lanzan en el *stream* actual de PyTorch, así
    que el orden respecto a las operaciones de torch está garantizado.

**Verificación** (`python kernels_cuda.py`): cada kernel se compara con su equivalente en
PyTorch FP32 (TF32 apagado para la comparación) y se mide con eventos CUDA, con lote de 512
(prueba del 15/11; repetir en la sesión final):

| Kernel | Grid | Bloque | Error máx. (relativo) | Propio (ms) | PyTorch (ms) |
|---|---|---|---|---|---|
| normalizar 64×64×3 | (1536, 4, 4) | (16, 16) | 6.0e-08 | 0.13 | 0.28 |
| conv_bn_relu 3→32, k5, 64×64 | (1024, 4, 4) | (16, 16) | 0 | 3.12 | 5.20 |
| conv_bn_relu 32→64, k3, 32×32 | (2048, 2, 2) | (16, 16) | 0 | 3.92 | 4.36 |
| conv_bn_relu 64→128, k3, 16×16 | (4096, 1, 1) | (16, 16) | 4.0e-06 | 4.01 | 2.49 |
| conv_bn_relu 128→128, k3, 16×16 | (4096, 1, 1) | (16, 16) | 6.9e-06 | 7.90 | 3.30 |
| conv_bn_relu 128→64, k3, 16×16 | (2048, 1, 1) | (16, 16) | 7.1e-06 | 4.43 | 1.96 |
| maxpool2 64×16×16 | (32768, 1, 1) | (16, 16) | 0 | 0.13 | 0.16 |
| densa 4096→256 + ReLU | (16, 32) | (16, 16) | 2.8e-06 | 0.59 | 0.13 |
| densa 256→29 | (2, 32) | (16, 16) | 2.8e-07 | 0.27 | 0.09 |
| argmax, 29 clases | (4,) | (128,) | 0 discrepancias | 0.15 | 0.01 |

Red completa: logits con error relativo de 1.5e-06 y **100 % de predicciones iguales** a las de
PyTorch. Forward por lote de 512: **25.4 ms** con nuestros kernels contra **19.5 ms** de cuDNN
con TF32.

Lectura: en las capas grandes y con pocos canales (64×64 y 32×32) nuestro kernel le gana a
cuDNN FP32, porque la memoria compartida y la fusión con BN y ReLU ahorran pasadas por
memoria. En las capas de 16×16 con 128 canales, cuDNN gana hasta 2.4×: usa Tensor Cores y
tiles más grandes elegidos por autoajuste. En las capas densas y el argmax, cuBLAS y PyTorch
ganan, aunque el costo absoluto es menor a 1 ms. Esto respalda la decisión de entrenar con
cuDNN y usar los kernels propios donde se puede diseñar y medir el reparto.

### 3.2 Entrenamiento: el reparto que eligen cuDNN y cuBLAS (`perfilar_gpu.py`)

El *backward* y el optimizador se dejan en PyTorch: reescribirlos a mano no cambia la
matemática y sí haría el entrenamiento distinto del de la línea base. El reparto de esos
kernels lo eligen las bibliotecas, así que se **midió** con `torch.profiler` en lugar de
suponerlo, con **Nsight Compute** (`ncu`, el perfilador oficial de NVIDIA).
`python perfilar_gpu.py` entrena 5 lotes, evalúa un lote de 512 con cuDNN y otro con nuestros
kernels, todo dentro de un rango NVTX, y `ncu` registra cada kernel lanzado (408 lanzamientos,
104 configuraciones distintas; tabla completa en `resultados/perfil/kernels.csv`).

- **Herramienta:** `torch.profiler` no sirve en esta PC. Su CUPTI 12.8 falla con
  `CUPTI_ERROR_INVALID_DEVICE` ante la RTX 5070 Laptop con el driver 596.49.
- **Permisos:** `ncu` requiere habilitar el acceso a los contadores de rendimiento de la GPU
  (*NVIDIA Control Panel → Developer → Manage GPU Performance Counters → Allow access to all
  users*).
- **Duraciones:** `ncu` ejecuta los kernels de a uno, con la caché vaciada y el reloj a
  frecuencia base. Sus duraciones sirven para comparar kernels entre sí, no como tiempo real.

Configuraciones representativas (prueba del 15/11):

| Kernel (biblioteca) | Operación | Grid | Bloque | Reg./hilo | Mem. compartida | Ocupación alcanzada |
|---|---|---|---|---|---|---|
| `sm80_xmma_fprop_implicit_gemm_tf32…` (cuDNN) | convolución forward en TF32 con Tensor Cores | 1×1024×1 | 128 | 242 | 66,560 B | 8.3 % |
| `cutlass__5x_cudnn::Kernel` (cuDNN/CUTLASS) | convolución (GEMM implícito) | 4096×1×1 / 512×1×1 | 128 | 150–248 | 74,752–82,944 B | 8.3 % |
| `implicit_convolve_sgemm` (cuDNN) | convolución FP32 sin Tensor Cores | 65536×1×1 / 8192×1×1 | 8×8 = 64 | 63–78 | 3,328 B | 48–66 % |
| `cutlass::Kernel2` (CUTLASS) | GEMM de las capas densas | 32×1×36 | 128 | 80 | 26,624 B | 24 % |
| `cudnn::bn_fw_tr_1C11_kernel_NCHW` | BatchNorm (entrenamiento) | 32×1×1 | 512 | 38 | 1,168 B | 33 % |
| `cudnn::bn_fw_inf_1C11_kernel_NCHW` | BatchNorm (inferencia) | 32×512×1 | 512 | 17 | 1,024 B | 83–86 % |
| `at::vectorized_elementwise_kernel` (PyTorch) | ReLU, conversión, escalado | 65536×1×1 / 8192×1×1 | 128 | 30 | 1,024 B | 86 % |
| **`k_conv_bn_relu` (nuestro)** | conv + BN + ReLU | 1024×4×4 … 4096×1×1 | 16×16 | 55–61 | 13,824 B | **66 %** |
| **`k_maxpool2` (nuestro)** | max-pool | 16384×2×2 / 32768×1×1 | 16×16 | 24 | 1,024 B | 73 % |
| **`k_densa` (nuestro)** | GEMM por tiles | 16×32×1 / 2×32×1 | 16×16 | 40 | 3,136 B | 85 % / 30 % |
| **`k_normalizar` (nuestro)** | uint8 NHWC → float NCHW | 1536×4×4 | 16×16 | 20 | 1,024 B | 77 % |
| **`k_argmax` (nuestro)** | argmax por imagen | 4×1×1 | 128 | 40 | 1,024 B | 8 % |

Qué muestra:

1. **El grid y el bloque medidos de nuestros kernels coinciden exactamente con los diseñados**
   (sección 3.1).
2. **Dos estrategias opuestas en la convolución.**
   - Las de cuDNN con Tensor Cores lanzan pocos bloques de 128 hilos con **mucho trabajo por
     hilo**: 242 registros y 66 KB de memoria compartida por bloque. Eso limita la ocupación a
     8 % (un bloque por SM), y la latencia se oculta con paralelismo dentro del hilo
     (*tiles* de 128×128 en registros) y no con muchos warps.
   - Nuestro kernel usa 16 salidas por hilo y 55 registros, y alcanza 66 % de ocupación.
   - Ganar con baja ocupación requiere *tiles* grandes y Tensor Cores, lo que explica por qué
     cuDNN nos gana en las capas de 128 canales.
3. **Las operaciones elemento a elemento** (ReLU, conversiones, BN en inferencia) usan grids 1D
   con un hilo por elemento (o por vector de 4) y bloques de 128 a 512 hilos. Con 86 % de
   ocupación están limitadas por el ancho de banda de memoria, no por el cálculo.
4. **En el entrenamiento hay 408 lanzamientos para 5 lotes**, unos 80 kernels por paso de
   Adam (forward, backward y optimizador). Con lotes de 64, muchos kernels son cortos y el
   costo fijo de lanzarlos pesa. Es uno de los factores que limitan el speedup (ver la discusión del informe).

Cómo leerlo:
- **Convoluciones → GEMM implícito.** cuDNN trata la convolución como una multiplicación de
  matrices con M = lote·alto·ancho, N = canales de salida y K = canales de entrada·k·k, sin
  materializar la matriz *im2col*. Cada bloque calcula un tile de M×N con Tensor Cores (TF32),
  y el grid es (⌈M/tile_M⌉, ⌈N/tile_N⌉).
- **ReLU, BN y Adam → elemento a elemento.** Usan grids 1D con bloques de 128 a 512 hilos, un
  hilo (o un vector de 4) por elemento.
- **Max-pool → un hilo por elemento de salida**, como nuestro `maxpool2`.

## 4. Resultados

**Sesión final de medición, 15/11/2026** (Core Ultra 9 275HX + RTX 5070 Laptop, Windows en
*Best performance*, conectada a la corriente, aplicaciones cerradas). Media de 3 corridas de
3 épocas por versión, con 5 min de pausa entre ejecuciones. Fuente: `resultados/tiempos.csv` →
`python resumen.py` → `resultados/resumen.csv`. Logs, `cpu.csv`, `gpu.csv` y capturas en
`resultados/evidencia/<version>/`.

| Fase | secuencial, 1 núcleo (s) | CPU 16 hilos (s) | **optimizado** (s) | optimizado_numba (s) | **speedup vs 1 núcleo** | speedup vs CPU 16 hilos |
|---|---|---|---|---|---|---|
| carga (CPU) | 26.85 ± 1.38 | 25.71 ± 0.18 | 6.68 ± 0.07 (4 procesos) | 6.87 ± 0.50 | **4.0×** | 3.8× |
| transferencia CPU→GPU | — | — | 0.10 ± 0.04 (258 MiB) | 0.13 ± 0.04 | — | — |
| entrenamiento (GPU) | 380.53 ± 4.18 | 54.43 ± 0.82 | 4.46 ± 0.05 | 4.53 ± 0.08 | **85.3×** | 12.2× |
| evaluación (GPU) | 15.91 ± 0.35 | 2.47 ± 0.07 | 0.24 ± 0.01 | 0.33 ± 0.00 | **66.6×** | 10.3× |
| **total** | **423.31 ± 4.06** | 82.64 ± 0.60 | **11.50 ± 0.10** | 11.88 ± 0.55 | **36.8×** | 7.2× |

| Versión | Accuracy test | F1 macro test |
|---|---|---|
| secuencial (1 hilo, determinista) | 99.61 % (las 3 corridas) | 99.60 % |
| CPU 16 hilos | 99.55 % (las 3 corridas) | 99.55 % |
| optimizado (GPU) | 99.58 / 99.64 / 99.64 % | 99.57 / 99.64 / 99.64 % |
| optimizado_numba (GPU, evaluación con kernels propios) | 99.61 / 99.73 / 99.64 % | 99.61 / 99.73 / 99.64 % |

Lectura:

1. **Entrenamiento: 85× contra 1 núcleo y 12× contra el mejor CPU multihilo.** Una época baja
   de ~127 s (1 núcleo) y ~18 s (16 hilos) a **1.45 s**.
   - El primer número es el que pide el enunciado. El segundo es la comparación honesta: el
     CPU de 24 núcleos tampoco es lento, y la ventaja real de la GPU viene de sus Tensor Cores
     (TF32) y de su ancho de banda de memoria.
2. **Total: 36.8×**, de 7 min a 11.5 s, pero muy lejos de 85×. Es la **ley de Amdahl**:
   - En la línea base la carga era el 6 % del tiempo. En la versión optimizada es el **58 %**
     (6.7 de 11.5 s).
   - La carga solo baja 4× porque abrir cada archivo no escala (ver el barrido de B en
     PLAN.md).
   - Aunque la GPU entrenara en 0 s, el total no bajaría de ~7 s.
3. **La transferencia es despreciable (0.1 s, el 0.9 % del total)** porque los datos viajan
   **una sola vez y en uint8** (258 MiB) y quedan en la memoria de la GPU para las 3 épocas. Si
   se copiara cada lote en float32, se moverían 4× más bytes en 723 copias pequeñas.
4. **El calentamiento importa:** en la corrida 0 (no registrada) la primera época tardó 6.6 s
   contra 1.45 s de las demás, por la creación del contexto CUDA, la carga de cuDNN y cuBLAS y
   el autoajuste de `cudnn.benchmark`.
5. **Evaluación con kernels propios:** 0.33 s contra 0.24 s con cuDNN (1.4× más lenta), con
   las mismas predicciones. Coincide con la medición por kernel de la sección 3.1.
6. **Uso de la GPU** (de `gpu.csv` y del Administrador de tareas, GPU 1 → "3D"): **96–99 %**
   durante el entrenamiento, hasta **113 W de 115 W**, 87 °C y ~0.8 GB de memoria dedicada.
   Con este driver (WDDM), el trabajo CUDA aparece en el motor "3D" del Administrador de
   tareas, porque no se ofrece un motor "Cuda" aparte. `nvidia-smi` lista `python.exe` como
   proceso de cómputo (tipo C).

## 5. Por qué el accuracy en GPU no es idéntico al de CPU

- **TF32:** las convoluciones usan una mantisa de 10 bits en lugar de 23. Cada producto se
  redondea distinto y, a lo largo de ~720 pasos de Adam, las trayectorias de los pesos
  divergen un poco.
- **cuDNN no es determinista** con `benchmark = True`: puede elegir algoritmos distintos y
  algunos usan sumas atómicas en el *backward*, cuyo orden varía.
- **La suma en punto flotante no es asociativa:** miles de hilos suman en otro orden que el
  `for` de la CPU.
- Por eso la CPU con 1 hilo es reproducible (99.61 % en las 3 corridas preliminares) y la GPU
  varía en décimas entre corridas. El criterio de aceptación es quedar a menos de 1 punto.
- **La evaluación con kernels propios** usa el mismo modelo entrenado y debe dar
  prácticamente las mismas predicciones que cuDNN. Es FP32 puro, y la diferencia con TF32
  solo cambia predicciones casi empatadas.
