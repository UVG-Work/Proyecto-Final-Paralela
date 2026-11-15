# Pasos 1 y 2 — Análisis y medición de la solución secuencial

Responsable: A. Código: `preparacion.py`, `red.py`, `secuencial.py`, `resumen.py`.

## 0. De dónde sale la solución secuencial

El "Laboratorio 3. Deep Learning: Reconocimiento de Lenguaje de Señas ASL" de Data Science
quedó en el repositorio `CC_3084_Laboratorio_3` con **solo el EDA** (`01_EDA_Lab3.ipynb`) y el
split resultante (`dataset_splits.csv`); el notebook de modelado no está ahí. La versión
secuencial de este proyecto reconstruye el pipeline siguiendo la metodología que ese EDA dejó
documentada (secciones 9 y 10). Decisiones y su origen:

| Decisión | Valor | Justificación |
|---|---|---|
| Datos | split del Lab 3: 22,022 imágenes, 29 clases (15,415 / 3,303 / 3,304) | El EDA ya aplicó la limpieza: descarte del 5 % menos nítido **por clase** (varianza del Laplaciano) y de imágenes sobreexpuestas (>30 % de píxeles quemados). Reusar el CSV garantiza los mismos datos en todas las versiones. |
| Resolución | 64×64 | Primera de las resoluciones propuestas (9.2). El EDA advierte que a 32×32 desaparece el detalle del pulgar que distingue M/N/S. |
| Redimensionamiento | Pillow, filtro `BOX` | Promedio de área: el equivalente de `INTER_AREA` que pide la metodología para evitar *aliasing*. |
| Modelo | `AlexNetASL`: AlexNet adaptada a 64×64 | Es el "modelo profundo de referencia" de la metodología (9.5). VGG, MLP y SVM quedan fuera: el proyecto trata de acelerar **un** entrenamiento, no de comparar modelos. |
| Normalización | BatchNorm después de cada convolución | Reemplazo moderno de la LRN de la AlexNet original. Ver "primera versión" abajo. |
| *Dropout* | ninguno | En 3 épocas no hay sobreajuste que evitar: con *dropout* 0.25 el test quedó en 85 %, sin él en 91–95 % (con lr constante). |
| Optimizador | Adam con OneCycle (lr máx. 1e-3) | Con lr constante el accuracy variaba 82–94 % entre corridas idénticas en GPU; con OneCycle, 99.5–99.7 % y estable. |
| Aumento de datos | ninguno | La metodología lo condiciona a que haya desbalance; el split quedó con 1.15 (max/min) y el propio EDA concluye que "no requiere sobremuestreo". |
| Métricas | accuracy y F1 macro | F1 macro por el desbalance residual (sección 10 del EDA). |
| Entrenamiento | lote 64, 3 épocas, semilla 42 | Suficiente para ~99.5 % en test y para que la línea base de 1 núcleo quepa en una sesión de medición (~15 min por corrida). |

**Primera versión, descartada.** La primera versión seguía a la AlexNet de 2012 al pie de la
letra: sin normalización, *dropout* 0.5 y lr constante. Se midió completa en CPU (3 corridas,
`resultados/preliminar/v1_sin_bn/`):

- Durante toda la primera época la pérdida se quedó en 3.368 = ln 29, es decir, la red
  adivinaba al azar entre las 29 clases.
- Al final de las 3 épocas llegó a **64 % en test**.

Las tres decisiones de la tabla (BatchNorm, sin *dropout*, OneCycle) salen de comparar
variantes en la GPU, donde cada prueba tarda segundos y no minutos.

## 1. Análisis de la solución secuencial

### 1.1 Partes principales del programa

| Parte | Dónde | Función |
|---|---|---|
| Configuración | `secuencial.py: main()` | Lee argumentos (`--corridas`, `--epocas`, `--hilos`), fija PyTorch a 1 hilo y el nombre de la versión. |
| Localización del dataset | `preparacion.ruta_dataset()` | Pide a kagglehub la copia local de `grassknoted/asl-alphabet` y ubica la carpeta con una subcarpeta por clase. |
| Lectura del split | `preparacion.leer_split()` | Lee el CSV del Lab 3 y devuelve, por conjunto, las rutas y las etiquetas numéricas (orden alfabético de clases). |
| Carga de imágenes | `preparacion.cargar_secuencial()` → `preparacion.cargar_imagen()` | Un `for` que, imagen por imagen, lee el JPG, lo decodifica, lo pasa a RGB y lo reduce a 64×64; al final apila todo en un arreglo. |
| Modelo | `red.AlexNetASL` | 5 bloques convolución + BatchNorm + ReLU, 3 *max-pooling*, clasificador de 2 capas densas. |
| Entrenamiento | `red.entrenar()` | Por época: baraja, arma lotes de 64, propaga hacia adelante, calcula la pérdida, retropropaga, actualiza pesos con Adam y ajusta el lr (OneCycle). |
| Evaluación | `red.evaluar()` | Predice validación y prueba en lotes de 512 y calcula accuracy y F1 macro. |
| Medición | `preparacion.registrar()`, `resumen.py` | Cronometra cada fase con `time.perf_counter()`, la agrega a `resultados/tiempos.csv` y luego la promedia. |

### 1.2 Operaciones, desde la recepción de las imágenes hasta el resultado

| No. | Operación identificada | Descripción de su funcionamiento |
|---|---|---|
| 1 | Lectura del split | Lee 22,022 filas `clase, ruta, conjunto` y convierte cada clase en un índice 0–28. |
| 2 | Lectura del archivo | Abre el JPG (200×200, ~12.4 KB en promedio) y lee sus bytes del disco o de la caché del sistema operativo. |
| 3 | Decodificación JPEG | Deshace la compresión (Huffman + transformada coseno inversa) y produce 200×200×3 bytes. |
| 4 | Conversión a RGB | Garantiza 3 canales aunque el archivo venga en otro modo. |
| 5 | Reducción a 64×64 | Cada píxel de salida es el promedio del área de ~3.1×3.1 píxeles de entrada que le corresponde (filtro `BOX`). |
| 6 | Apilado | Junta las imágenes en un arreglo `N×64×64×3` uint8 (12,288 B por imagen; 270.6 MB en total). |
| 7 | Barajado | Al inicio de cada época, una permutación aleatoria (semilla fija) de los 15,415 índices de entrenamiento. |
| 8 | Armado del lote | Toma 64 imágenes según la permutación, las pasa a float32, reordena a `N×C×H×W` y divide entre 255. |
| 9 | Propagación hacia adelante | Convolución → BatchNorm → ReLU (×5, con *max-pool* tras la 1.ª, 2.ª y 5.ª), aplanado, densa → ReLU → densa: produce 29 *logits* por imagen. |
| 10 | Cálculo de la pérdida | *Softmax* + entropía cruzada contra la etiqueta real, promediada en el lote. |
| 11 | Retropropagación | Regla de la cadena capa por capa, de la salida a la entrada: gradiente respecto a cada peso y a cada activación. |
| 12 | Actualización de pesos | Adam: actualiza los 1,372,861 parámetros con sus promedios móviles del gradiente y del gradiente al cuadrado; OneCycle ajusta el lr en cada paso. |
| 13 | Inferencia | Propagación hacia adelante sin gradientes sobre validación y prueba; la clase predicha es el `argmax` de los *logits*. |
| 14 | Métricas | Compara predicciones con etiquetas: accuracy y F1 macro. |

### 1.3 Operaciones que se ejecutan repetidamente

Conteos para **una corrida** (3 épocas). Se usan multiplicaciones-suma (MAC); 1 MAC = 2 FLOP.

| Operación repetitiva | Datos que procesa | Cantidad de repeticiones |
|---|---|---|
| Carga de una imagen (ops. 2–5) | un JPG 200×200×3 → 64×64×3 uint8 | **22,022** |
| Paso de entrenamiento (ops. 8–12) | un lote de 64 imágenes de 3×64×64 float32 (3 MB) | 241 lotes por época (240 de 64 + 1 de 55) × 3 = **723** |
| Convolución (dentro de 9 y 11) | ventanas 5×5 o 3×3 sobre los mapas de activación | 104.2 M MAC por imagen hacia adelante; ×~3 con retropropagación → **≈1.46 × 10¹³ MAC** (29 TFLOP) en el entrenamiento |
| BatchNorm | cada salida de las 5 convoluciones (normaliza con media y varianza del lote por canal) | 278,528 por imagen hacia adelante → ≈1.3 × 10¹⁰ en el entrenamiento |
| ReLU | cada activación de las 5 convoluciones y la primera densa | 278,784 por imagen hacia adelante → ≈1.3 × 10¹⁰ en el entrenamiento |
| *Max-pooling* 2×2 | ventanas de 4 valores | 53,248 por imagen → ≈2.5 × 10⁹ en el entrenamiento |
| Actualización Adam (op. 12) | 1,372,861 parámetros | 723 actualizaciones → ≈9.9 × 10⁸ parámetros actualizados |
| Inferencia (op. 13) | lotes de 512 imágenes | 14 lotes (7 val + 7 test), 6,607 imágenes → ≈7.0 × 10¹¹ MAC |

Desglose por capa (de aquí salen los 104.2 M; C lo necesita para el diseño en GPU, porque cada
convolución se ejecuta como una multiplicación de matrices):

| Capa | Entrada | Salida | Parámetros | MAC por imagen |
|---|---|---|---|---|
| conv1 5×5 + BN + ReLU + pool | 3×64×64 | 32×32×32 | 2,464 | 9,830,400 |
| conv2 3×3 + BN + ReLU + pool | 32×32×32 | 64×16×16 | 18,560 | 18,874,368 |
| conv3 3×3 + BN + ReLU | 64×16×16 | 128×16×16 | 73,984 | 18,874,368 |
| conv4 3×3 + BN + ReLU | 128×16×16 | 128×16×16 | 147,712 | 37,748,736 |
| conv5 3×3 + BN + ReLU + pool | 128×16×16 | 64×8×8 | 73,856 | 18,874,368 |
| densa 1 + ReLU | 4,096 | 256 | 1,048,832 | 1,048,576 |
| densa 2 | 256 | 29 | 7,453 | 7,424 |
| **Total** | | | **1,372,861** | **105,258,240** |

(Parámetros de cada convolución = pesos sin sesgo + 2 por canal de BatchNorm. El total
coincide con `sum(p.numel() for p in AlexNetASL(29).parameters())`.)

Lo que muestra la tabla: el 99 % de la aritmética está en las convoluciones, y cada una es
la misma operación aplicada a miles de posiciones y canales independientes. La carga de
imágenes, en cambio, son 22,022 tareas independientes pero cada una es E/S + decodificación
con mucha ramificación. Esa diferencia es la base del paso 3.

## 2. Medición del tiempo de ejecución secuencial

### 2.1 Procedimiento

- **Hardware:** Intel Core i9-14900HX (8 núcleos P + 16 núcleos E, 32 hilos), 32 GB RAM,
  NVIDIA RTX 5070 Laptop 8 GB (no se usa en esta versión). Windows 11.
- **Software:** Python 3.10.11, PyTorch 2.11.0+cu128, Pillow, NumPy, scikit-learn.
- **Secuencial de verdad:** `torch.set_num_threads(1)` y `set_num_interop_threads(1)`; la carga
  es un `for` en el hilo principal.
- **Comando:** `python secuencial.py` → 3 corridas × (carga + 3 épocas + evaluación).
- **Calentamiento sin cronometrar** antes de la corrida 1: se leen los bytes de los 22,022
  archivos (caché de disco caliente) y se da un paso de entrenamiento con datos falsos
  (inicializaciones perezosas de PyTorch). Así las 3 corridas parten de las mismas condiciones
  y todas cuentan para el promedio.
- **Cronómetro:** `time.perf_counter()` alrededor de cada fase; cada fase es una fila en
  `resultados/tiempos.csv`. `python resumen.py` calcula media, desviación, mínimo y máximo.
- **Evidencia:** uso de CPU por núcleo cada 2 s con
  `typeperf "\Processor(*)\% Processor Time"`.

### 2.2 Resultados (preliminares)

> **Preliminares.** Se tomaron el 14/11/2026 con la laptop **ocupada** por otras aplicaciones
> (Roblox Studio, Blender, navegador, Discord). Sirven como orden de magnitud y para validar
> el pipeline completo; el informe usa los de la sesión final (PLAN.md, sección 5). Archivos:
> `resultados/preliminar/` (`tiempos.csv`, `resumen.csv`, `log_secuencial.txt`,
> `cpu_por_nucleo_secuencial.csv`).

Por corrida (segundos; el calentamiento de 9.0 s no se cuenta):

| Corrida | Carga | Entrenamiento (3 épocas) | Evaluación | Total | CPU usada por **otros** procesos |
|---|---|---|---|---|---|
| 1 | 18.86 | 712.62 | 34.29 | 765.79 | 25.0 % |
| 2 | 39.09 | 768.86 | 41.37 | 849.35 | 35.7 % |
| 3 | 48.29 | 994.33 | 46.47 | 1089.16 | 38.6 % |
| **Media** | **35.41** | **825.27** | **40.71** | **901.43** | |
| Desv. estándar | 15.06 | 149.09 | 6.12 | 167.86 | |
| % del total | 3.9 % | **91.6 %** | 4.5 % | 100 % | |

(La última columna es el uso total de CPU de la máquina registrado por `typeperf`; nuestro
proceso de 1 hilo aporta 3.1 % de ese número, 1 de 32 procesadores lógicos.)

Resultado del modelo, **idéntico en las tres corridas**: la pérdida por época es la misma
(1.6654 → 0.2381 → 0.0340) y el accuracy también. Con 1 hilo y semilla fija la ejecución es
determinista.

| Conjunto | Accuracy | F1 macro |
|---|---|---|
| Validación (3,303) | 99.52 % | 99.51 % |
| Prueba (3,304) | 99.61 % | 99.60 % |

Rendimiento por unidad, tomando la mejor corrida (la menos perturbada):

| Fase | Por unidad | Rendimiento |
|---|---|---|
| Carga | 0.86 ms por imagen | 1,168 imágenes/s |
| Entrenamiento | 237.5 s por época, 15.4 ms por imagen | ~41 GFLOP/s (0.63 GFLOP por imagen entrenada) |
| Evaluación | 5.2 ms por imagen | ~40 GFLOP/s |

Lo que dicen estos números para los pasos siguientes:

- **El entrenamiento es el 91.6 % del tiempo**, y por eso es el blanco de CUDA. Por la ley de
  Amdahl, aunque el entrenamiento costara cero, con la carga y la evaluación secuenciales el
  total no bajaría de 76.1 s (35.4 + 40.7): un speedup total máximo de 901.4 / 76.1 ≈ 11.8x.
- Por eso importa también la fase de B. Si la GPU reduce el entrenamiento a segundos, la
  carga secuencial (35 s) pasa a ser la mayor parte de lo que queda.
- **Orientación para C** (prueba exploratoria de A en la misma máquina ocupada, no la medición
  oficial): en la GPU una época tomó ~1.7 s contra ~240 s aquí, con 99.5–99.7 % en test.

### 2.3 Factores que influyen en el rendimiento

1. **Carga de otros procesos.** Es el factor dominante en esta medición. El tiempo por
   corrida creció 766 → 849 → 1089 s junto con el uso de CPU ajeno (25 % → 36 % → 39 %), con
   el mismo código y los mismos datos. La desviación estándar del total es 18.6 % de la media,
   inaceptable para el informe.
2. **Temperatura.** En una laptop la CPU y la GPU comparten disipador. Bajo carga sostenida el
   procesador baja su frecuencia de *turbo*, y la GPU marcaba 84 °C al terminar. Cada corrida
   hereda el calor de la anterior, así que el sesgo cae siempre sobre las últimas.
3. **Núcleos P y E.** El i9-14900HX mezcla 8 núcleos de rendimiento y 16 de eficiencia, y es
   Windows el que decide en cuál corre el hilo. La época de 219.7 s de la corrida 1 contra las
   de 325–338 s de la corrida 3 (+50 %) es consistente con frecuencia reducida o con el hilo
   movido a un núcleo E. No se puede distinguir cuál con estos datos.
4. **Caché de disco.** En la prueba inicial, con la caché del sistema operativo fría, la carga
   costó 2.6 ms por imagen; ya caliente, 1.0 ms (2.6x). Por eso el calentamiento lee los
   22,022 archivos antes de cronometrar: la fase de carga mide decodificación y reducción
   (CPU), no la velocidad del disco ni el escaneo del antivirus en la primera lectura. El
   dataset vive en `~/.cache/kagglehub`, fuera de OneDrive, para que la sincronización no
   interfiera.
5. **Eficiencia aritmética.** ~41 GFLOP/s es alrededor de una cuarta parte del pico teórico de
   un núcleo P con AVX2 (32 FLOP por ciclo a ~5 GHz ≈ 160 GFLOP/s). Dos causas:
   - BatchNorm, ReLU y *max-pool* hacen pocas operaciones por byte leído; las limita la
     memoria, no el cálculo.
   - La retropropagación reordena datos además de multiplicar.
6. **Formato de memoria.** En este CPU, entrenar con los datos en orden *channels_last*
   (NHWC) es 15 % más rápido que en NCHW (968 contra 1134 ms por lote de 64). La versión
   secuencial usa el más rápido, para no inflar el speedup de la GPU con una línea base
   debilitada.
7. **Tamaño de lote.** Con lotes de 64 cada multiplicación de matrices es mediana. Lotes más
   grandes aprovecharían mejor los vectores del CPU (y mucho más la GPU), pero cambiarían el
   entrenamiento, así que el lote se fijó igual para ambas versiones.
