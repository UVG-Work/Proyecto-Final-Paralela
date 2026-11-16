"""Kernels CUDA propios escritos en CUDA Python (numba-cuda): inferencia de AlexNetASL.

Paso 4 y 5 del enunciado: aqui el reparto del trabajo en hilos, bloques y grids
lo DISENAMOS nosotros (en el entrenamiento con PyTorch lo decide cuDNN). Este
modulo reimplementa el forward de la red entrenada con 5 kernels:

  kernel                 un hilo calcula            bloque          grid
  ---------------------  -------------------------  --------------  -----------------------------
  normalizar             1 pixel de 1 canal          16x16 = 256     (N*C, ceil(H/16), ceil(W/16))
  conv_bn_relu           1 pixel x 16 canales        16x16 = 256     (N*Co/16, ceil(H/16), ceil(W/16))
  maxpool2               1 pixel de salida (n,c)     16x16 = 256     (N*C, ceil(Ho/16), ceil(Wo/16))
  densa                  1 elemento de Y = XW^T+b    16x16 = 256     (ceil(O/16), ceil(N/16))
  argmax                 1 imagen                    128             ceil(N/128)

Decisiones (detalle en docs/paso4_5_gpu.md):
- (n, canal) va en grid.x porque gridDim.y/z tienen tope de 65,535 y con lotes
  de 512 y 128 canales hay 65,536 planos; gridDim.x admite 2^31-1.
- Bloques de 256 hilos (8 warps): multiplo de 32 y con suficientes bloques
  residentes por SM para ocultar la latencia de memoria global.
- threadIdx.x recorre la coordenada x de la imagen, que es la contigua en
  memoria (NCHW): los 32 hilos de un warp leen/escriben direcciones
  consecutivas (acceso coalescido).
- conv_bn_relu: el bloque copia a memoria compartida el parche de entrada que
  necesita ((16+K-1)^2 valores por canal, de 4 canales a la vez) y los pesos de
  sus 16 filtros. Cada valor del parche se reutiliza K*K veces (vecinos) x 16
  veces (canales de salida, acumulados en registros) en vez de leerse de la
  memoria global cada vez. Todos los hilos de un bloque leen el mismo peso a la
  vez (difusion desde memoria compartida).
- BatchNorm en inferencia es afin por canal, asi que se FUSIONA en los pesos de
  la convolucion (w' = w*g/sqrt(var+eps), b' = beta - media*g/sqrt(var+eps)) y
  la ReLU se aplica antes de escribir: una sola pasada por memoria en vez de 3.
- densa: GEMM por tiles de 16x16 en memoria compartida; el tile de W se guarda
  transpuesto y con una columna de relleno (16x17) para evitar conflictos de
  banco al escribirlo.
- Toda la aritmetica es float32 (las GPU de consumo hacen FP64 a 1/64 de
  velocidad); los literales se escriben como float32(...) para que numba no
  promueva a float64.

Los tensores de PyTorch se pasan a numba sin copias via __cuda_array_interface__
y los kernels se lanzan en el stream actual de PyTorch, asi el orden entre
operaciones de torch y kernels propios queda garantizado.

Uso:  python kernels_cuda.py        # verifica cada kernel contra PyTorch y mide su tiempo
"""
import math

import numpy as np
import torch
from numba import cuda
from numpy import float32
from torch import nn

TB = 16          # lado del bloque 2D para kernels de imagen
TG = 16          # lado del tile de la GEMM
K_MAX = 5        # kernel de convolucion mas grande de la red (memoria compartida estatica)
HILOS_1D = 128
COB = 16         # canales de salida por hilo en la convolucion (bloqueo en registros)
CIB = 4          # canales de entrada por etapa de memoria compartida
# Formas de memoria compartida: deben ser constantes de compilacion, no expresiones.
LADO_PARCHE = TB + K_MAX - 1   # parche de entrada con halo para el K mas grande
TG_PAD = TG + 1                # columna de relleno contra conflictos de banco
KK_MAX = K_MAX * K_MAX         # pesos por filtro y canal, para el K mas grande


# ---------------------------------------------------------------------------
# Kernels
# ---------------------------------------------------------------------------
@cuda.jit
def k_normalizar(x, y):
    """uint8 NHWC -> float32 NCHW en [0, 1] (lo mismo que red.a_tensor)."""
    nc = cuda.blockIdx.x
    C = y.shape[1]
    n = nc // C
    c = nc % C
    py = cuda.blockIdx.y * TB + cuda.threadIdx.y
    px = cuda.blockIdx.z * TB + cuda.threadIdx.x
    if py < y.shape[2] and px < y.shape[3]:
        y[n, c, py, px] = float32(x[n, py, px, c]) / float32(255.0)


def _hacer_conv_bn_relu(K):
    """Fabrica del kernel de convolucion para un tamano de filtro K fijo.

    K (y COB, CIB) quedan como constantes de compilacion: el compilador
    desenrolla los ciclos internos y el acumulador acc[COB] vive en registros.
    """
    T = TB + K - 1                  # lado del parche con halo
    PAD = K // 2
    TT = T * T
    KK = K * K
    HILOS = TB * TB

    @cuda.jit
    def k_conv_bn_relu(x, w, b, y):
        """Convolucion 'same' (stride 1) + BatchNorm fusionado + ReLU.

        x (N, Ci, H, W), w (Co, Ci, K, K), b (Co), y (N, Co, H, W).
        Bloque (bx, by, bz): imagen n y grupo de COB canales de salida (bx),
        tile 16x16 de pixeles (by, bz). Cada hilo calcula SU pixel en los COB
        canales del grupo.
        """
        parche = cuda.shared.array((CIB, LADO_PARCHE, LADO_PARCHE), float32)
        pesos = cuda.shared.array((CIB, KK_MAX, COB), float32)
        Co = y.shape[1]
        Ci = x.shape[1]
        H = x.shape[2]
        W = x.shape[3]
        grupos = (Co + COB - 1) // COB
        n = cuda.blockIdx.x // grupos
        co0 = (cuda.blockIdx.x % grupos) * COB
        tx = cuda.threadIdx.x
        ty = cuda.threadIdx.y
        hilo = ty * TB + tx
        y0 = cuda.blockIdx.y * TB
        x0 = cuda.blockIdx.z * TB

        acc = cuda.local.array(COB, float32)
        for j in range(COB):
            acc[j] = float32(0.0)

        for c0 in range(0, Ci, CIB):
            # 1) Carga cooperativa a memoria compartida del parche de CIB canales
            #    de entrada (con halo y ceros fuera de la imagen)...
            i = hilo
            while i < CIB * TT:
                cc = i // TT
                r = i - cc * TT
                ly = r // T
                lx = r - ly * T
                gy = y0 + ly - PAD
                gx = x0 + lx - PAD
                v = float32(0.0)
                if c0 + cc < Ci and gy >= 0 and gy < H and gx >= 0 and gx < W:
                    v = x[n, c0 + cc, gy, gx]
                parche[cc, ly, lx] = v
                i += HILOS
            # ...y de los pesos de los COB filtros del grupo para esos canales.
            i = hilo
            while i < CIB * KK * COB:
                cc = i // (KK * COB)
                r = i - cc * (KK * COB)
                kk = r // COB
                j = r - kk * COB
                v = float32(0.0)
                if c0 + cc < Ci and co0 + j < Co:
                    v = w[co0 + j, c0 + cc, kk // K, kk % K]
                pesos[cc, kk, j] = v
                i += HILOS
            cuda.syncthreads()
            # 2) Cada valor del parche se lee UNA vez y se reutiliza en COB
            #    multiplicaciones; los pesos son iguales para todo el bloque
            #    (difusion desde memoria compartida).
            for cc in range(CIB):
                for ky in range(K):
                    for kx in range(K):
                        v = parche[cc, ty + ky, tx + kx]
                        for j in range(COB):
                            acc[j] += v * pesos[cc, ky * K + kx, j]
            cuda.syncthreads()          # nadie sobrescribe mientras otro lee

        oy = y0 + ty
        ox = x0 + tx
        if oy < H and ox < W:
            for j in range(COB):
                if co0 + j < Co:
                    v = acc[j] + b[co0 + j]
                    y[n, co0 + j, oy, ox] = v if v > float32(0.0) else float32(0.0)

    return k_conv_bn_relu


_K_CONV = {k: _hacer_conv_bn_relu(k) for k in (3, 5)}


@cuda.jit
def k_maxpool2(x, y):
    """Max-pool 2x2 con stride 2. x (N, C, H, W) -> y (N, C, H/2, W/2)."""
    nc = cuda.blockIdx.x
    C = y.shape[1]
    n = nc // C
    c = nc % C
    oy = cuda.blockIdx.y * TB + cuda.threadIdx.y
    ox = cuda.blockIdx.z * TB + cuda.threadIdx.x
    if oy < y.shape[2] and ox < y.shape[3]:
        iy = 2 * oy
        ix = 2 * ox
        y[n, c, oy, ox] = max(max(x[n, c, iy, ix], x[n, c, iy, ix + 1]),
                              max(x[n, c, iy + 1, ix], x[n, c, iy + 1, ix + 1]))


@cuda.jit
def k_densa(x, w, b, y, relu):
    """Y = X W^T + b (y ReLU opcional), GEMM por tiles en memoria compartida.

    x (N, I), w (O, I) como nn.Linear, b (O), y (N, O).
    Hilo (ty, tx) del bloque (bx, by) calcula y[by*16+ty, bx*16+tx].
    """
    sx = cuda.shared.array((TG, TG), float32)
    sw = cuda.shared.array((TG, TG_PAD), float32)    # +1: sin conflictos de banco
    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y
    N = x.shape[0]
    I = x.shape[1]
    O = w.shape[0]
    fila = cuda.blockIdx.y * TG + ty     # imagen
    col = cuda.blockIdx.x * TG + tx      # neurona de salida
    fila_w = cuda.blockIdx.x * TG + ty   # neurona cuya fila de W carga este hilo
    acc = float32(0.0)
    for t in range((I + TG - 1) // TG):
        k = t * TG + tx                  # tx recorre I: lecturas coalescidas en x y en w
        sx[ty, tx] = x[fila, k] if (fila < N and k < I) else float32(0.0)
        sw[tx, ty] = w[fila_w, k] if (fila_w < O and k < I) else float32(0.0)
        cuda.syncthreads()
        for kk in range(TG):
            acc += sx[ty, kk] * sw[kk, tx]
        cuda.syncthreads()
    if fila < N and col < O:
        v = acc + b[col]
        if relu and v < float32(0.0):
            v = float32(0.0)
        y[fila, col] = v


@cuda.jit
def k_argmax(x, out):
    """Clase con mayor logit por imagen. x (N, clases) -> out (N) int64."""
    n = cuda.grid(1)
    if n < x.shape[0]:
        mejor = 0
        vmax = x[n, 0]
        for j in range(1, x.shape[1]):
            if x[n, j] > vmax:
                vmax = x[n, j]
                mejor = j
        out[n] = mejor


# ---------------------------------------------------------------------------
# Lanzadores: calculan grid/bloque y reservan la salida con el allocator de torch
# ---------------------------------------------------------------------------
def _nb(t):
    """Vista numba (sin copia) de un tensor CUDA de torch."""
    return cuda.as_cuda_array(t.detach())


def _stream():
    return cuda.external_stream(torch.cuda.current_stream().cuda_stream)


def _grid_img(planos, H, W):
    return (planos, math.ceil(H / TB), math.ceil(W / TB)), (TB, TB)


def normalizar(x_u8):
    N, H, W, C = x_u8.shape
    y = torch.empty((N, C, H, W), device=x_u8.device, dtype=torch.float32)
    grid, bloque = _grid_img(N * C, H, W)
    k_normalizar[grid, bloque, _stream()](_nb(x_u8.contiguous()), _nb(y))
    return y


def grid_conv(N, Co, H, W):
    return _grid_img(N * math.ceil(Co / COB), H, W)


def conv_bn_relu(x, w, b):
    N, _, H, W = x.shape
    Co, _, K, _ = w.shape
    y = torch.empty((N, Co, H, W), device=x.device, dtype=torch.float32)
    grid, bloque = grid_conv(N, Co, H, W)
    _K_CONV[K][grid, bloque, _stream()](_nb(x), _nb(w), _nb(b), _nb(y))
    return y


def maxpool2(x):
    N, C, H, W = x.shape
    y = torch.empty((N, C, H // 2, W // 2), device=x.device, dtype=torch.float32)
    grid, bloque = _grid_img(N * C, H // 2, W // 2)
    k_maxpool2[grid, bloque, _stream()](_nb(x), _nb(y))
    return y


def densa(x, w, b, relu):
    N = x.shape[0]
    O = w.shape[0]
    y = torch.empty((N, O), device=x.device, dtype=torch.float32)
    grid = (math.ceil(O / TG), math.ceil(N / TG))
    k_densa[grid, (TG, TG), _stream()](_nb(x), _nb(w), _nb(b), _nb(y), relu)
    return y


def argmax(x):
    out = torch.empty(x.shape[0], device=x.device, dtype=torch.int64)
    k_argmax[math.ceil(x.shape[0] / HILOS_1D), HILOS_1D, _stream()](_nb(x), _nb(out))
    return out


# ---------------------------------------------------------------------------
# Motor de inferencia: recorre el modelo entrenado y lo traduce a los kernels
# ---------------------------------------------------------------------------
class InferenciaNumba:
    """Forward de un AlexNetASL entrenado, ejecutado solo con los kernels de arriba."""

    def __init__(self, modelo):
        self.capas = []
        mods = [m for m in list(modelo.features) + list(modelo.clasificador)]
        i = 0
        with torch.no_grad():
            while i < len(mods):
                m = mods[i]
                if isinstance(m, nn.Conv2d):
                    bn = mods[i + 1]
                    assert isinstance(bn, nn.BatchNorm2d) and isinstance(mods[i + 2], nn.ReLU)
                    assert m.stride == (1, 1) and m.padding[0] == m.kernel_size[0] // 2
                    escala = bn.weight / torch.sqrt(bn.running_var + bn.eps)
                    w = (m.weight * escala[:, None, None, None]).float().contiguous()
                    b = (bn.bias - bn.running_mean * escala).float().contiguous()
                    if m.bias is not None:
                        b = b + m.bias * escala
                    self.capas.append(("conv", w, b))
                    i += 3
                elif isinstance(m, nn.MaxPool2d):
                    self.capas.append(("pool",))
                    i += 1
                elif isinstance(m, nn.Flatten):
                    self.capas.append(("flatten",))
                    i += 1
                elif isinstance(m, nn.Linear):
                    relu = i + 1 < len(mods) and isinstance(mods[i + 1], nn.ReLU)
                    self.capas.append(("densa", m.weight.float().contiguous(),
                                       m.bias.float().contiguous(), relu))
                    i += 2 if relu else 1
                else:
                    raise TypeError(f"capa no soportada: {m}")

    def logits(self, x_u8):
        x = normalizar(x_u8)
        for capa in self.capas:
            if capa[0] == "conv":
                x = conv_bn_relu(x, capa[1], capa[2])
            elif capa[0] == "pool":
                x = maxpool2(x)
            elif capa[0] == "flatten":
                x = x.reshape(x.shape[0], -1)       # NCHW contiguo: vista, sin copia
            else:
                x = densa(x, capa[1], capa[2], capa[3])
        return x

    def predecir(self, x_u8):
        return argmax(self.logits(x_u8))


@torch.no_grad()
def evaluar(modelo, X, y, lote=512):
    """Igual que red.evaluar (accuracy, F1 macro), pero el forward usa kernels propios."""
    from sklearn.metrics import f1_score
    motor = InferenciaNumba(modelo.eval())
    pred = torch.cat([motor.predecir(X[i:i + lote]).cpu()
                      for i in range(0, len(X), lote)]).numpy()
    real = y.cpu().numpy()
    return float((pred == real).mean()), float(f1_score(real, pred, average="macro"))


# ---------------------------------------------------------------------------
# Verificacion y microbenchmark (python kernels_cuda.py)
# ---------------------------------------------------------------------------
def _medir(f, repeticiones=20):
    """ms por llamada con eventos CUDA (miden tiempo de GPU, no de encolado)."""
    f()
    torch.cuda.synchronize()
    ini = torch.cuda.Event(enable_timing=True)
    fin = torch.cuda.Event(enable_timing=True)
    ini.record()
    for _ in range(repeticiones):
        f()
    fin.record()
    torch.cuda.synchronize()
    return ini.elapsed_time(fin) / repeticiones


def _verificar():
    import torch.nn.functional as F
    import red as R

    # Comparar contra FP32 exacto, no TF32.
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.benchmark = True
    dev = "cuda"
    g = torch.Generator(device=dev).manual_seed(0)
    N = 512

    print(f"GPU: {torch.cuda.get_device_name(0)}, numba-cuda, lote = {N}\n")
    print(f"{'kernel':34s} {'grid':>20s} {'bloque':>8s} {'err. max':>10s} "
          f"{'propio ms':>10s} {'torch ms':>9s}")

    def fila(nombre, grid, bloque, err, t_nb, t_t):
        print(f"{nombre:34s} {str(grid):>20s} {str(bloque):>8s} {err:10.2e} "
              f"{t_nb:10.3f} {t_t:9.3f}")

    x_u8 = torch.randint(0, 256, (N, 64, 64, 3), dtype=torch.uint8, device=dev, generator=g)
    ref = R.a_tensor(x_u8, dev)
    out = normalizar(x_u8)
    fila("normalizar 64x64x3", _grid_img(N * 3, 64, 64)[0], (TB, TB),
         (out - ref).abs().max().item(), _medir(lambda: normalizar(x_u8)),
         _medir(lambda: R.a_tensor(x_u8, dev)))

    for ci, co, k, hw in [(3, 32, 5, 64), (32, 64, 3, 32), (64, 128, 3, 16),
                          (128, 128, 3, 16), (128, 64, 3, 16)]:
        x = torch.rand((N, ci, hw, hw), device=dev, generator=g)
        w = torch.randn((co, ci, k, k), device=dev, generator=g) * 0.1
        b = torch.randn(co, device=dev, generator=g)
        ref = F.relu(F.conv2d(x, w, b, padding=k // 2))
        out = conv_bn_relu(x, w, b)
        fila(f"conv_bn_relu {ci}->{co} k{k} {hw}x{hw}", grid_conv(N, co, hw, hw)[0],
             (TB, TB), ((out - ref).abs().max() / ref.abs().max()).item(),
             _medir(lambda: conv_bn_relu(x, w, b)),
             _medir(lambda: F.relu(F.conv2d(x, w, b, padding=k // 2))))

    x = torch.rand((N, 64, 16, 16), device=dev, generator=g)
    fila("maxpool2 64x16x16", _grid_img(N * 64, 8, 8)[0], (TB, TB),
         (maxpool2(x) - F.max_pool2d(x, 2)).abs().max().item(),
         _medir(lambda: maxpool2(x)), _medir(lambda: F.max_pool2d(x, 2)))

    for i_, o_, relu in [(4096, 256, True), (256, 29, False)]:
        x = torch.rand((N, i_), device=dev, generator=g)
        w = torch.randn((o_, i_), device=dev, generator=g) * 0.05
        b = torch.randn(o_, device=dev, generator=g)
        f_ref = (lambda: F.relu(F.linear(x, w, b))) if relu else (lambda: F.linear(x, w, b))
        ref = f_ref()
        out = densa(x, w, b, relu)
        fila(f"densa {i_}->{o_}{' +relu' if relu else ''}",
             (math.ceil(o_ / TG), math.ceil(N / TG)), (TG, TG),
             ((out - ref).abs().max() / ref.abs().max()).item(),
             _medir(lambda: densa(x, w, b, relu)), _medir(f_ref))

    x = torch.randn((N, 29), device=dev, generator=g)
    fila("argmax 29 clases", (math.ceil(N / HILOS_1D),), (HILOS_1D,),
         float((argmax(x) != x.argmax(1)).sum().item()),
         _medir(lambda: argmax(x)), _medir(lambda: x.argmax(1)))

    # Red completa con pesos aleatorios en modo eval (BN con estadisticas no triviales).
    torch.manual_seed(R.SEMILLA)
    modelo = R.AlexNetASL(29).to(dev)
    with torch.no_grad():
        for m in modelo.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.running_mean.uniform_(-0.5, 0.5)
                m.running_var.uniform_(0.5, 2.0)
                m.weight.uniform_(0.5, 1.5)
                m.bias.uniform_(-0.2, 0.2)
    modelo.eval()
    motor = InferenciaNumba(modelo)
    with torch.no_grad():
        ref = modelo(R.a_tensor(x_u8, dev))
        out = motor.logits(x_u8)
        coinciden = (out.argmax(1) == ref.argmax(1)).float().mean().item()
        print(f"\nRed completa (lote {N}): err. rel. max de logits = "
              f"{((out - ref).abs().max() / ref.abs().max()).item():.2e}, "
              f"predicciones iguales = {coinciden * 100:.2f} %")
        t_nb = _medir(lambda: motor.logits(x_u8))
        torch.backends.cudnn.allow_tf32 = True
        t_t = _medir(lambda: modelo(R.a_tensor(x_u8, dev)))
        print(f"forward por lote de {N}: kernels propios {t_nb:.2f} ms, "
              f"PyTorch/cuDNN (TF32) {t_t:.2f} ms")


if __name__ == "__main__":
    _verificar()
