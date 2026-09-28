"""
Configuracion comun de la tarea "Desbloqueo facial".

Todos los scripts (paso1, paso2, paso3, extra y desbloquear) importan este
archivo, asi las rutas y los hiperparametros se cambian en un solo lugar.

Estructura de carpetas esperada (dentro de esta carpeta):

    datos/
        tensorflow_datasets/         <- CelebA descargada por tfds (automatico)
        celeba/                      <- (solo si FUENTE_CELEBA = "local")
            img_align_celeba/        <- las ~200k imagenes .jpg de CelebA
            list_attr_celeba.txt     <- (o .csv) los 40 atributos por imagen
            list_eval_partition.txt  <- (o .csv, opcional) particion train/val/test
        mi_rostro/                   <- fotos de TU rostro (recortadas a la cara)
        otros_rostros/               <- (opcional) fotos de otras personas
    modelos/                         <- aqui se guardan los modelos entrenados
    figuras/                         <- aqui se guardan las graficas

Las carpetas de datos y de salida se pueden redirigir con las variables de
entorno DESBLOQUEO_DATOS y DESBLOQUEO_SALIDA (util para pruebas).
"""

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Rutas
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent

DATOS_DIR = Path(os.environ.get("DESBLOQUEO_DATOS", BASE_DIR / "datos"))
SALIDA_DIR = Path(os.environ.get("DESBLOQUEO_SALIDA", BASE_DIR))

# De donde sale CelebA:
#   "tfds"  -> tensorflow_datasets: tfds.load("celeb_a"). Se descarga sola la
#              primera vez (~1.4 GB) y se guarda en TFDS_DIR.
#   "local" -> archivos descargados a mano en CELEBA_DIR (ver datos.py).
FUENTE_CELEBA = os.environ.get("DESBLOQUEO_FUENTE_CELEBA", "tfds")
TFDS_DIR = DATOS_DIR / "tensorflow_datasets"

CELEBA_DIR = DATOS_DIR / "celeba"
CELEBA_IMGS = CELEBA_DIR / "img_align_celeba"
MI_ROSTRO_DIR = DATOS_DIR / "mi_rostro"
OTROS_DIR = DATOS_DIR / "otros_rostros"

MODELOS_DIR = SALIDA_DIR / "modelos"
FIG_DIR = SALIDA_DIR / "figuras"

# Archivos que produce cada paso
MODELO_CELEBA = MODELOS_DIR / "cnn_celeba.keras"          # paso 1
MODELO_ETAPA2 = MODELOS_DIR / "rostro_etapa2.keras"       # paso 2
MODELO_FINAL = MODELOS_DIR / "rostro_final.keras"         # paso 3
UMBRAL_JSON = MODELOS_DIR / "umbral.json"                 # paso 2 / paso 3
ATRIBUTOS_JSON = MODELOS_DIR / "atributos_celeba.json"    # paso 1

# ---------------------------------------------------------------------------
# Imagenes
# ---------------------------------------------------------------------------
# Las imagenes se recortan a un cuadrado central y se redimensionan a
# IMG_SIZE x IMG_SIZE. 128 es un buen compromiso entre detalle y velocidad
# (en CPU se puede bajar a 96 o 64 para entrenar mas rapido).
IMG_SIZE = 128
CANALES = 3
BATCH = 64
SEMILLA = 42

# ---------------------------------------------------------------------------
# Paso 1: pre-entrenamiento con CelebA (40 atributos)
# ---------------------------------------------------------------------------
# CelebA tiene ~163k imagenes de entrenamiento. Entrenar con todas en CPU es
# lento, asi que por defecto se usa un subconjunto. None = usar todas.
CELEBA_MAX_TRAIN = 40000
CELEBA_MAX_VAL = 5000
CELEBA_MAX_TEST = 5000
EPOCAS_CELEBA = 15
LR_CELEBA = 1e-3

# Filtros de cada bloque convolucional de la red (4 bloques -> la imagen de
# 128x128 termina en un mapa de 8x8 antes del Global Average Pooling).
FILTROS = (32, 64, 128, 256)
DENSA_CELEBA = 256          # neuronas de la capa densa del clasificador CelebA

# ---------------------------------------------------------------------------
# Paso 2 y 3: clasificador "es mi rostro / no es mi rostro"
# ---------------------------------------------------------------------------
FRAC_VAL_ROSTRO = 0.2       # fraccion de MIS fotos reservada para validacion
REPETICIONES_POS = 20       # cada foto mia se repite (con aumento distinto)
NEG_POR_POS_VAL = 10        # negativos por cada positivo en validacion
DENSAS_ROSTRO = (128,)      # capas densas del nuevo clasificador (1 o 2)
DROPOUT_ROSTRO = 0.4

EPOCAS_ETAPA2 = 20
LR_ETAPA2 = 1e-3

EPOCAS_ETAPA3 = 15
LR_ETAPA3 = 1e-5            # LR pequeno: solo "refinar" lo ya aprendido
BLOQUES_A_DESCONGELAR = 1   # cuantos bloques conv finales se re-entrenan

# Para desbloquear un telefono importa mas no dejar pasar a otra persona
# (falsa aceptacion) que rechazar al duenio de vez en cuando. Se elige el
# umbral que deja la tasa de falsa aceptacion (FAR) por debajo de este valor.
FAR_OBJETIVO = 0.01


def crear_carpetas_salida():
    """Crea modelos/ y figuras/ si no existen."""
    MODELOS_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
