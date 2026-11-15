"""FASE 2 - la red neuronal: modelo, entrenamiento y evaluacion (PyTorch).

Las funciones reciben el dispositivo como parametro ("cpu" o "cuda"): la
version secuencial y la optimizada ejecutan exactamente las mismas
operaciones, solo cambia donde. No modificar el modelo ni los
hiperparametros para acelerar una sola de las dos versiones.
"""
import time

import torch
from sklearn.metrics import f1_score
from torch import nn

SEMILLA = 42      # la misma que uso el EDA para el split
LOTE = 64
LR = 1e-3


def _conv(c_in, c_out, k):
    """Convolucion + BatchNorm + ReLU. BN hace innecesario el sesgo de la convolucion."""
    return [nn.Conv2d(c_in, c_out, k, padding=k // 2, bias=False),
            nn.BatchNorm2d(c_out), nn.ReLU(inplace=True)]


class AlexNetASL(nn.Module):
    """AlexNet reducida para entradas de 64x64 (metodologia del Lab 3, 9.5).

    Misma estructura que AlexNet (5 convoluciones con ReLU, 3 max-pool y un
    clasificador denso), con menos canales y kernels mas chicos porque la
    entrada es 64x64 y no 224x224. ~105 M multiplicaciones-suma por imagen.

    Dos cambios respecto a la AlexNet de 2012, medidos con 3 epocas (ver
    docs/paso1_2_secuencial.md, seccion 0):
    - BatchNorm en lugar de LRN, su reemplazo moderno. Sin normalizacion la red
      pasa la primera epoca adivinando al azar (perdida = ln 29) y llega a 64-70 %.
    - Sin dropout: en 3 epocas no hay sobreajuste que evitar y solo frena el
      aprendizaje (85 % con dropout 0.25 contra 91-95 % sin el).
    """

    def __init__(self, n_clases):
        super().__init__()
        self.features = nn.Sequential(
            *_conv(3, 32, 5), nn.MaxPool2d(2),      # 32x32
            *_conv(32, 64, 3), nn.MaxPool2d(2),     # 16x16
            *_conv(64, 128, 3),
            *_conv(128, 128, 3),
            *_conv(128, 64, 3), nn.MaxPool2d(2),    # 8x8
        )
        self.clasificador = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 256), nn.ReLU(inplace=True),
            nn.Linear(256, n_clases),
        )

    def forward(self, x):
        return self.clasificador(self.features(x))


def a_tensor(x_uint8, dispositivo):
    """Lote uint8 NHWC -> float32 NCHW en [0, 1], en el dispositivo dado.

    Se convierte DESPUES de mover: a la GPU viaja uint8 (4x menos bytes que
    float32) y la division por 255 corre en la GPU.

    El formato de memoria es el mejor de cada dispositivo (medido, ms por lote
    de 64 en entrenamiento): permute() deja los datos en orden channels_last,
    que en CPU es 15 % mas rapido (968 contra 1134 ms); en GPU BatchNorm con
    channels_last es 3.3x mas lento (27.5 contra 8.4 ms), asi que ahi se copia
    a NCHW contiguo. La matematica es identica; solo cambia el orden en memoria.
    """
    x = x_uint8.to(dispositivo, non_blocking=True).permute(0, 3, 1, 2).float().div_(255)
    return x.contiguous() if x.is_cuda else x


def sincronizar(dispositivo):
    """Las llamadas CUDA son asincronas: sin esto el cronometro mide solo el encolado."""
    if torch.device(dispositivo).type == "cuda":
        torch.cuda.synchronize()


def entrenar(modelo, X, y, dispositivo, epocas, log=print):
    """Adam + entropia cruzada. X uint8 NHWC e y int64, tensores en el mismo dispositivo
    (CPU o el de entrenamiento). Devuelve los segundos de cada epoca.

    El orden de los lotes sale de un generador CPU con semilla fija, asi que es
    el mismo en cualquier dispositivo.
    """
    modelo.to(dispositivo)
    n = len(X)
    opt = torch.optim.Adam(modelo.parameters(), lr=LR)
    # OneCycle: sube el lr hasta LR y lo baja casi a 0 al final. Con lr constante
    # los pesos finales "rebotan" y el accuracy en GPU variaba 82-94 % entre corridas.
    plan_lr = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=LR, total_steps=epocas * -(-n // LOTE))
    criterio = nn.CrossEntropyLoss()
    gen = torch.Generator().manual_seed(SEMILLA)
    tiempos = []
    for ep in range(1, epocas + 1):
        modelo.train()
        t0 = time.perf_counter()
        orden = torch.randperm(n, generator=gen).to(X.device)
        suma = torch.zeros((), device=dispositivo)
        for i in range(0, n, LOTE):
            idx = orden[i:i + LOTE]
            xb = a_tensor(X[idx], dispositivo)
            yb = y[idx].to(dispositivo, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            perdida = criterio(modelo(xb), yb)
            perdida.backward()
            opt.step()
            plan_lr.step()
            # Acumular en tensor y no con .item(): .item() obliga a la CPU a
            # esperar a la GPU en cada lote.
            suma += perdida.detach() * len(idx)
        sincronizar(dispositivo)
        tiempos.append(time.perf_counter() - t0)
        log(f"  epoca {ep}/{epocas}: perdida={suma.item() / n:.4f}  {tiempos[-1]:.1f} s")
    return tiempos


def calentar(n_clases, dispositivo):
    """Un paso de entrenamiento y una inferencia descartables (no se cronometran).

    Dispara las inicializaciones perezosas que si no caerian dentro de la corrida
    1: imports internos del optimizador y oneDNN en CPU; contexto CUDA, carga de
    cuBLAS/cuDNN y su autoajuste en GPU.
    """
    modelo = AlexNetASL(n_clases)
    X = torch.zeros(LOTE, 64, 64, 3, dtype=torch.uint8)
    y = torch.zeros(LOTE, dtype=torch.int64)
    entrenar(modelo, X, y, dispositivo, 1, log=lambda *_: None)
    with torch.no_grad():
        modelo.eval()(a_tensor(X, dispositivo))
    sincronizar(dispositivo)


@torch.no_grad()
def evaluar(modelo, X, y, dispositivo, lote=512):
    """(accuracy, F1 macro). F1 macro porque el split quedo con un desbalance
    residual de 1.15 (seccion 10 del EDA)."""
    modelo.eval()
    pred = torch.cat([modelo(a_tensor(X[i:i + lote], dispositivo)).argmax(1).cpu()
                      for i in range(0, len(X), lote)]).numpy()
    real = y.cpu().numpy()
    return float((pred == real).mean()), float(f1_score(real, pred, average="macro"))
