"""
Carga de datos para la tarea de desbloqueo facial.

Aqui vive todo lo relacionado con leer imagenes y convertirlas en
`tf.data.Dataset` listos para entrenar:

1. CelebA (paso 1): imagen -> vector de 40 atributos binarios
   (Smiling, Eyeglasses, Male, Young, ...). Es un problema *multi-etiqueta*:
   una misma cara puede tener varios atributos a la vez.

   Por defecto CelebA se obtiene con TensorFlow Datasets:
       tfds.load("celeb_a", split="train")
   La primera vez descarga ~1.4 GB y lo prepara en datos/tensorflow_datasets/;
   las siguientes veces lo lee de ahi. Cada ejemplo es un diccionario con
   "image" (218x178x3 uint8), "attributes" (40 booleanos), "landmarks" e
   "identity". Aqui solo se usan "image" y "attributes".

   Alternativa (config.FUENTE_CELEBA = "local"): leer los archivos de CelebA
   ya descargados a mano en datos/celeba/ (formato oficial .txt o .csv de
   Kaggle). Util si la descarga de TFDS falla (los archivos estan en Google
   Drive y a veces se excede la cuota de descargas).

2. Mi rostro (pasos 2 y 3): imagen -> 1 si es mi cara, 0 si es otra persona.
   Los negativos ("rostros de los demas") salen de datos/otros_rostros/ y de
   la particion de prueba de CelebA, asi el conjunto queda equilibrado aunque
   yo tenga pocas fotos.

3. Aumento de datos sintetico. El enunciado sugiere la clase
   `ImageDataGenerator` de Keras, pero en Keras 3 (la version instalada) esa
   clase ya no existe. Su reemplazo oficial son las *capas de preprocesamiento*
   (RandomFlip, RandomRotation, RandomZoom, ...), que hacen exactamente lo
   mismo: generar en cada epoca versiones distintas (giradas, desplazadas,
   con otro brillo) de la misma foto.
"""

import csv
import random
from pathlib import Path

import numpy as np
import tensorflow as tf
import keras
from keras import layers

import config

AUTOTUNE = tf.data.AUTOTUNE
EXTENSIONES = {".jpg", ".jpeg", ".png", ".bmp"}

# Nombres de las particiones en TFDS (0 = train, 1 = val, 2 = test)
SPLITS_TFDS = {0: "train", 1: "validation", 2: "test"}

# Buffer para barajar CelebA de TFDS (no cabe entera en memoria)
BUFFER_BARAJEO = 2000


# ===========================================================================
# Preprocesamiento de una imagen
# ===========================================================================
def recortar_cuadrado(img):
    """Recorta el cuadrado central de la imagen.

    Las imagenes alineadas de CelebA miden 178x218 (ancho x alto) y la cara
    esta centrada, asi que al quedarnos con el cuadrado central 178x178 no se
    pierde la cara y evitamos deformarla al redimensionar. A mis fotos se les
    aplica el mismo recorte para que ambas fuentes se vean igual.
    """
    alto = tf.shape(img)[0]
    ancho = tf.shape(img)[1]
    lado = tf.minimum(alto, ancho)
    return tf.image.crop_to_bounding_box(
        img, (alto - lado) // 2, (ancho - lado) // 2, lado, lado)


def preprocesar(img):
    """uint8 (alto, ancho, 3) -> float32 (IMG_SIZE, IMG_SIZE, 3) en [0, 1].

    Recorte cuadrado -> redimensionar -> escalar a [0, 1]. Se usa igual para
    CelebA (TFDS o local), mis fotos y las fotos de desbloquear.py.
    """
    img = recortar_cuadrado(img)
    img = tf.image.resize(img, (config.IMG_SIZE, config.IMG_SIZE),
                          antialias=True)
    img = tf.clip_by_value(tf.cast(img, tf.float32) / 255.0, 0.0, 1.0)
    img.set_shape((config.IMG_SIZE, config.IMG_SIZE, config.CANALES))
    return img


def cargar_imagen(ruta):
    """Lee un archivo de imagen (jpg/png/bmp) y lo preprocesa."""
    datos = tf.io.read_file(ruta)
    img = tf.io.decode_image(datos, channels=config.CANALES,
                             expand_animations=False)
    return preprocesar(img)


def listar_imagenes(carpeta):
    """Lista (ordenada) de rutas de imagenes dentro de una carpeta."""
    carpeta = Path(carpeta)
    if not carpeta.exists():
        return []
    return sorted(str(p) for p in carpeta.rglob("*")
                  if p.suffix.lower() in EXTENSIONES)


# ===========================================================================
# Aumento de datos (reemplazo de ImageDataGenerator)
# ===========================================================================
def construir_aumento(intenso=True):
    """Capas de aumento de datos aleatorio.

    Equivalencia con los argumentos de ImageDataGenerator:
        horizontal_flip=True        -> RandomFlip("horizontal")
        rotation_range=18           -> RandomRotation(0.05)  (0.05 * 360 = 18)
        zoom_range=0.15             -> RandomZoom(0.15)
        width/height_shift_range    -> RandomTranslation(0.1, 0.1)
        brightness_range            -> RandomBrightness
        (no existia)                -> RandomContrast

    `intenso=False` se usa con CelebA, que ya tiene mucha variedad: basta con
    el espejo horizontal (que no cambia ninguno de los 40 atributos).
    """
    if not intenso:
        return keras.Sequential([layers.RandomFlip("horizontal")],
                                name="aumento_ligero")
    return keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.05, fill_mode="reflect"),
        layers.RandomZoom(0.15, fill_mode="reflect"),
        layers.RandomTranslation(0.1, 0.1, fill_mode="reflect"),
        layers.RandomBrightness(0.2, value_range=(0.0, 1.0)),
        layers.RandomContrast(0.3),
    ], name="aumento")


def _aplicar_aumento(aumento):
    """Devuelve una funcion para `ds.map` que aumenta un lote de imagenes."""
    def funcion(x, y):
        x = aumento(x, training=True)
        return tf.clip_by_value(x, 0.0, 1.0), y
    return funcion


# ===========================================================================
# CelebA desde TensorFlow Datasets
# ===========================================================================
def _importar_tfds():
    try:
        import tensorflow_datasets as tfds
    except ImportError as e:
        raise SystemExit("Falta tensorflow-datasets: "
                         "pip install tensorflow-datasets") from e
    return tfds


def _nombres_atributos_tfds():
    """Nombres de los 40 atributos, en el orden que se usa como salida."""
    tfds = _importar_tfds()
    info = tfds.builder("celeb_a", data_dir=str(config.TFDS_DIR)).info
    return list(info.features["attributes"].keys())


def _celeba_tfds(particion, max_imagenes=None, barajar=False):
    """(dataset sin lotes de (imagen, atributos), nombres) desde TFDS.

    `split="train[:40000]"` es la sintaxis de TFDS para tomar solo los
    primeros 40000 ejemplos de la particion (ver tensorflow.org/datasets/splits).
    """
    tfds = _importar_tfds()
    split = SPLITS_TFDS[particion]
    if max_imagenes is not None:
        split = f"{split}[:{max_imagenes}]"
    ds = tfds.load("celeb_a", split=split, data_dir=str(config.TFDS_DIR),
                   shuffle_files=barajar, download=True)
    nombres = _nombres_atributos_tfds()

    if barajar:
        ds = ds.shuffle(BUFFER_BARAJEO, seed=config.SEMILLA,
                        reshuffle_each_iteration=True)

    def convertir(ejemplo):
        # Diccionario de 40 booleanos -> vector float32 (40,) con 0 / 1
        atributos = tf.stack([tf.cast(ejemplo["attributes"][n], tf.float32)
                              for n in nombres])
        return preprocesar(ejemplo["image"]), atributos

    return ds.map(convertir, num_parallel_calls=AUTOTUNE), nombres


# ===========================================================================
# CelebA desde archivos locales (alternativa)
# ===========================================================================
def _mensaje_celeba_faltante():
    return (
        f"No se encontro CelebA en {config.CELEBA_DIR}.\n"
        "Con FUENTE_CELEBA = 'local' hay que descargarla a mano (pagina "
        "oficial https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html o Kaggle "
        "https://www.kaggle.com/datasets/jessicali9530/celeba-dataset) y dejar:\n"
        f"  {config.CELEBA_IMGS}\\*.jpg\n"
        f"  {config.CELEBA_DIR}\\list_attr_celeba.txt  (o .csv)\n"
        f"  {config.CELEBA_DIR}\\list_eval_partition.txt  (o .csv, opcional)\n"
        "O usar FUENTE_CELEBA = 'tfds' para descargarla automaticamente."
    )


def leer_atributos_celeba():
    """Lee el archivo de atributos de CelebA (modo local).

    Acepta los dos formatos que circulan:
      * list_attr_celeba.txt (oficial): linea 1 = numero de imagenes,
        linea 2 = nombres de los 40 atributos, despues
        "000001.jpg -1 1 1 -1 ..." separado por espacios.
      * list_attr_celeba.csv (Kaggle): encabezado "image_id,5_o_Clock_Shadow,..."

    Los valores vienen como -1 / 1; se convierten a 0 / 1 para poder usar
    sigmoide + entropia cruzada binaria.

    Regresa: (lista de nombres de archivo, matriz (N, 40) float32,
              lista con los nombres de los atributos)
    """
    ruta_txt = config.CELEBA_DIR / "list_attr_celeba.txt"
    ruta_csv = config.CELEBA_DIR / "list_attr_celeba.csv"
    archivos, valores = [], []

    if ruta_txt.exists():
        with open(ruta_txt, encoding="utf-8") as f:
            lineas = f.read().splitlines()
        nombres_attr = lineas[1].split()
        for linea in lineas[2:]:
            partes = linea.split()
            if partes:
                archivos.append(partes[0])
                valores.append([int(v) for v in partes[1:]])
    elif ruta_csv.exists():
        with open(ruta_csv, newline="", encoding="utf-8") as f:
            lector = csv.reader(f)
            nombres_attr = next(lector)[1:]
            for fila in lector:
                if fila:
                    archivos.append(fila[0])
                    valores.append([int(v) for v in fila[1:]])
    else:
        raise FileNotFoundError(_mensaje_celeba_faltante())

    atributos = (np.array(valores, dtype=np.int8) > 0).astype(np.float32)
    return archivos, atributos, nombres_attr


def particion_celeba(archivos):
    """Particion oficial de CelebA: 0 = entrenamiento, 1 = validacion, 2 = prueba.

    Si no esta el archivo list_eval_partition, se hace una particion
    aleatoria 80 / 10 / 10 reproducible (con semilla fija).
    """
    ruta_txt = config.CELEBA_DIR / "list_eval_partition.txt"
    ruta_csv = config.CELEBA_DIR / "list_eval_partition.csv"
    mapa = {}
    if ruta_txt.exists():
        with open(ruta_txt, encoding="utf-8") as f:
            for linea in f:
                partes = linea.split()
                if len(partes) == 2:
                    mapa[partes[0]] = int(partes[1])
    elif ruta_csv.exists():
        with open(ruta_csv, newline="", encoding="utf-8") as f:
            lector = csv.reader(f)
            next(lector)
            for fila in lector:
                if len(fila) == 2:
                    mapa[fila[0]] = int(fila[1])

    if mapa:
        return np.array([mapa.get(a, 0) for a in archivos], dtype=np.int8)

    rng = np.random.default_rng(config.SEMILLA)
    u = rng.random(len(archivos))
    return np.where(u < 0.8, 0, np.where(u < 0.9, 1, 2)).astype(np.int8)


def _celeba_local(particion, max_imagenes=None, barajar=False):
    """(dataset sin lotes de (imagen, atributos), nombres) desde disco."""
    archivos, atributos, nombres = leer_atributos_celeba()
    part = particion_celeba(archivos)
    indices = np.where(part == particion)[0]
    if max_imagenes is not None and len(indices) > max_imagenes:
        rng = np.random.default_rng(config.SEMILLA + particion)
        indices = np.sort(rng.choice(indices, max_imagenes, replace=False))

    rutas = [str(config.CELEBA_IMGS / archivos[i]) for i in indices]
    ds = tf.data.Dataset.from_tensor_slices((rutas, atributos[indices]))
    if barajar:
        # Aqui si se puede barajar todo: solo se barajan rutas (texto)
        ds = ds.shuffle(len(rutas), seed=config.SEMILLA,
                        reshuffle_each_iteration=True)
    return (ds.map(lambda r, y: (cargar_imagen(r), y),
                   num_parallel_calls=AUTOTUNE), nombres)


# ===========================================================================
# CelebA: interfaz comun (elige TFDS o local segun config)
# ===========================================================================
def celeba(particion, max_imagenes=None, barajar=False):
    """Dataset SIN lotes de (imagen preprocesada, 40 atributos) y nombres."""
    if config.FUENTE_CELEBA == "tfds":
        return _celeba_tfds(particion, max_imagenes, barajar)
    return _celeba_local(particion, max_imagenes, barajar)


def dataset_celeba(particion, max_imagenes=None, entrenamiento=False):
    """Dataset de CelebA en lotes, listo para `model.fit` (paso 1).

    particion: 0 = train, 1 = val, 2 = test.
    max_imagenes: limita el numero de imagenes.
    entrenamiento: si es True se barajan los datos en cada epoca y se aplica
                   aumento ligero (espejo horizontal).

    Regresa (dataset, nombres_de_atributos, numero_de_imagenes).
    """
    ds, nombres = celeba(particion, max_imagenes, barajar=entrenamiento)
    n = int(ds.cardinality())
    ds = ds.batch(config.BATCH)
    if entrenamiento:
        ds = ds.map(_aplicar_aumento(construir_aumento(intenso=False)),
                    num_parallel_calls=AUTOTUNE)
    return ds.prefetch(AUTOTUNE), nombres, n


# ===========================================================================
# Mi rostro vs. los demas
# ===========================================================================
def dividir_mi_rostro():
    """Separa mis fotos en entrenamiento y validacion (por archivo).

    Es importante separar por archivo ANTES de aumentar: si una foto aparece
    en entrenamiento y una version girada de la misma foto en validacion, la
    validacion seria demasiado optimista.
    """
    rutas = listar_imagenes(config.MI_ROSTRO_DIR)
    if len(rutas) < 2:
        raise FileNotFoundError(
            f"Se necesitan fotos de tu rostro en {config.MI_ROSTRO_DIR} "
            "(al menos 2; se recomiendan 30 o mas, con distinta luz, "
            "expresion y angulo). Puedes tomarlas con capturar_fotos.py.")
    if len(rutas) < 20:
        print(f"[aviso] Solo hay {len(rutas)} fotos tuyas; con menos de 20 "
              "la validacion sera poco confiable.")
    rng = random.Random(config.SEMILLA)
    rng.shuffle(rutas)
    n_val = max(1, round(len(rutas) * config.FRAC_VAL_ROSTRO))
    return rutas[n_val:], rutas[:n_val]


def _a_uint8(img):
    """float [0,1] -> uint8, para guardar muchas imagenes en poca memoria."""
    return np.round(np.asarray(img) * 255).astype(np.uint8)


def leer_imagenes(rutas):
    """Carga una lista de archivos como arreglo uint8 (N, IMG, IMG, 3)."""
    if not rutas:
        return np.zeros((0, config.IMG_SIZE, config.IMG_SIZE, config.CANALES),
                        np.uint8)
    return np.stack([_a_uint8(cargar_imagen(r)) for r in rutas])


def imagenes_negativas(n_necesarios):
    """Rostros de OTRAS personas (la clase 0) como arreglo uint8.

    Primero se usan las fotos de datos/otros_rostros/ (si existen: amigos,
    familiares... son los negativos mas utiles porque se tomaron con la misma
    camara e iluminacion que mis fotos). El resto se completa con CelebA,
    usando la particion de prueba (y si no alcanza, la de validacion) para no
    reutilizar las imagenes con las que se pre-entreno la red.
    """
    otros = listar_imagenes(config.OTROS_DIR)
    random.Random(config.SEMILLA).shuffle(otros)
    otros = otros[:n_necesarios]
    lotes = [leer_imagenes(otros)]

    faltan = n_necesarios - len(otros)
    for particion in (2, 1):
        if faltan <= 0:
            break
        ds, _ = celeba(particion, max_imagenes=faltan)
        imgs = np.stack([_a_uint8(x) for x, _ in ds.as_numpy_iterator()])
        lotes.append(imgs)
        faltan -= len(imgs)

    negativos = np.concatenate(lotes)
    if len(negativos) < n_necesarios:
        print(f"[aviso] Solo hay {len(negativos)} negativos de "
              f"{n_necesarios} pedidos.")
    rng = np.random.default_rng(config.SEMILLA)
    return negativos[rng.permutation(len(negativos))][:n_necesarios]


def _dataset_binario(imagenes, etiquetas, entrenamiento):
    """Dataset en lotes a partir de arreglos uint8 ya cargados en memoria."""
    ds = tf.data.Dataset.from_tensor_slices(
        (imagenes, np.array(etiquetas, dtype=np.float32).reshape(-1, 1)))
    if entrenamiento:
        ds = ds.shuffle(len(imagenes), seed=config.SEMILLA,
                        reshuffle_each_iteration=True)
    ds = ds.map(lambda x, y: (tf.cast(x, tf.float32) / 255.0, y),
                num_parallel_calls=AUTOTUNE)
    ds = ds.batch(config.BATCH)
    if entrenamiento:
        ds = ds.map(_aplicar_aumento(construir_aumento(intenso=True)),
                    num_parallel_calls=AUTOTUNE)
    return ds.prefetch(AUTOTUNE)


def datasets_mi_rostro():
    """Construye los datasets equilibrados de entrenamiento y validacion.

    Entrenamiento:
        positivos = mis fotos de entrenamiento repetidas REPETICIONES_POS
                    veces. Como el aumento es aleatorio, cada repeticion es
                    una version sintetica distinta (girada, con zoom, etc.).
        negativos = el mismo numero de rostros de otras personas.
        -> 50% / 50%: la red no puede "hacer trampa" diciendo siempre "no".

    Validacion (sin aumento):
        positivos = mis fotos reservadas.
        negativos = NEG_POR_POS_VAL por cada positivo, para medir mejor la
                    tasa de falsa aceptacion (que otra persona desbloquee).

    Todas las imagenes se guardan en memoria como uint8 (128x128x3 = 48 KB
    cada una), asi que unos miles de imagenes ocupan pocos cientos de MB.

    Regresa (ds_train, ds_val, resumen) donde resumen es un dict con conteos.
    """
    rutas_train, rutas_val = dividir_mi_rostro()
    pos_train = np.repeat(leer_imagenes(rutas_train), config.REPETICIONES_POS,
                          axis=0)
    pos_val = leer_imagenes(rutas_val)

    n_neg_val = len(pos_val) * config.NEG_POR_POS_VAL
    negativos = imagenes_negativas(len(pos_train) + n_neg_val)
    neg_val, neg_train = negativos[:n_neg_val], negativos[n_neg_val:]

    x_train = np.concatenate([pos_train, neg_train])
    y_train = [1] * len(pos_train) + [0] * len(neg_train)
    x_val = np.concatenate([pos_val, neg_val])
    y_val = [1] * len(pos_val) + [0] * len(neg_val)

    resumen = {
        "fotos_mias_train": len(rutas_train),
        "fotos_mias_val": len(rutas_val),
        "positivos_train": len(pos_train),
        "negativos_train": len(neg_train),
        "positivos_val": len(pos_val),
        "negativos_val": len(neg_val),
    }
    return (_dataset_binario(x_train, y_train, entrenamiento=True),
            _dataset_binario(x_val, y_val, entrenamiento=False),
            resumen)
