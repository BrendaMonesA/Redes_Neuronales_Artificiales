"""
Arquitecturas y utilidades de transfer learning.

Idea central (Razavian et al., 2014, "CNN Features off-the-shelf"): las capas
convolucionales de una red entrenada en una tarea grande aprenden
*caracteristicas genericas* (bordes -> texturas -> partes -> rostros) que
sirven para otras tareas parecidas. Por eso separamos la red en dos piezas:

    imagen -> [ BASE CONVOLUCIONAL ] -> vector de caracteristicas -> [ CLASIFICADOR ]
              (se reutiliza)                                         (se reemplaza)

* En el paso 1 el clasificador predice los 40 atributos de CelebA.
* En el paso 2 se tira ese clasificador y se pone uno nuevo con UNA neurona
  de salida: "es mi rostro" (1) o "no lo es" (0).

La base convolucional se construye como un sub-modelo de Keras con nombre
"base_convolucional". Asi, "quitar el clasificador" es simplemente
`modelo.get_layer("base_convolucional")`, y congelar la parte pre-entrenada
es `base.trainable = False`.
"""

import keras
from keras import layers

import config

NOMBRE_BASE = "base_convolucional"


# ===========================================================================
# Base convolucional (extractor de caracteristicas)
# ===========================================================================
def bloque_conv(x, filtros, nombre):
    """Bloque tipo VGG: (Conv 3x3 -> BatchNorm -> ReLU) x 2 -> MaxPool 2x2.

    * Dos convoluciones 3x3 seguidas ven un campo receptivo de 5x5 con menos
      parametros que una sola 5x5.
    * BatchNormalization normaliza las activaciones de cada lote: acelera y
      estabiliza el entrenamiento. (Por eso la convolucion no lleva bias: la
      BN ya tiene su propio desplazamiento beta.)
    * MaxPooling reduce a la mitad el alto y ancho: cada bloque "ve" la cara
      a una escala mas gruesa y con mas filtros (mas abstraccion).

    Todas las capas llevan el prefijo del bloque (bloque1_, bloque2_, ...)
    para poder descongelar bloques completos en el paso 3.
    """
    for i in (1, 2):
        x = layers.Conv2D(filtros, 3, padding="same", use_bias=False,
                          name=f"{nombre}_conv{i}")(x)
        x = layers.BatchNormalization(name=f"{nombre}_bn{i}")(x)
        x = layers.Activation("relu", name=f"{nombre}_relu{i}")(x)
    return layers.MaxPooling2D(2, name=f"{nombre}_pool")(x)


def construir_base_convolucional():
    """Extractor de caracteristicas: imagen (128,128,3) -> vector (256,).

    Despues de los 4 bloques la imagen queda como un mapa 8x8x256.
    GlobalAveragePooling promedia cada uno de los 256 mapas, dando un vector
    de 256 numeros que resume "que rasgos hay en la cara". Este vector es el
    equivalente a la representacion de 4096 dimensiones que usa el articulo
    (capa 22 de OverFeat), pero mas pequeno porque la red es mas chica.
    """
    entrada = keras.Input(
        shape=(config.IMG_SIZE, config.IMG_SIZE, config.CANALES), name="imagen")
    x = entrada
    for i, filtros in enumerate(config.FILTROS, start=1):
        x = bloque_conv(x, filtros, f"bloque{i}")
    x = layers.GlobalAveragePooling2D(name="caracteristicas")(x)
    return keras.Model(entrada, x, name=NOMBRE_BASE)


# ===========================================================================
# Paso 1: red para los 40 atributos de CelebA
# ===========================================================================
def construir_cnn_celeba(n_atributos=40):
    """Base convolucional + clasificador denso multi-etiqueta.

    La salida tiene 40 neuronas con activacion SIGMOIDE (no softmax): cada
    neurona responde de forma independiente "tiene / no tiene" su atributo,
    porque una cara puede ser a la vez "Smiling", "Young" y "Eyeglasses".
    La perdida correspondiente es la entropia cruzada binaria promediada
    sobre los 40 atributos.
    """
    base = construir_base_convolucional()
    entrada = keras.Input(
        shape=(config.IMG_SIZE, config.IMG_SIZE, config.CANALES), name="imagen")
    x = base(entrada)
    x = layers.Dense(config.DENSA_CELEBA, activation="relu",
                     name="celeba_densa")(x)
    x = layers.Dropout(0.3, name="celeba_dropout")(x)
    salida = layers.Dense(n_atributos, activation="sigmoid",
                          name="atributos")(x)
    return keras.Model(entrada, salida, name="cnn_celeba")


# ===========================================================================
# Pasos 2 y 3: clasificador "mi rostro / no"
# ===========================================================================
def extraer_base(modelo):
    """Quita el clasificador de un modelo entrenado y regresa solo la base."""
    return modelo.get_layer(NOMBRE_BASE)


def construir_clasificador_rostro(base):
    """Base pre-entrenada + nuevo clasificador con 1 neurona de salida.

    El nuevo clasificador tiene 1 o 2 capas densas (config.DENSAS_ROSTRO)
    con Dropout, y una neurona final con sigmoide que da la probabilidad de
    que la foto sea mia. Esas capas se inicializan al azar; la base conserva
    los pesos aprendidos con CelebA.
    """
    entrada = keras.Input(
        shape=(config.IMG_SIZE, config.IMG_SIZE, config.CANALES), name="imagen")
    x = base(entrada)
    for i, unidades in enumerate(config.DENSAS_ROSTRO, start=1):
        x = layers.Dense(unidades, activation="relu",
                         name=f"rostro_densa{i}")(x)
        x = layers.Dropout(config.DROPOUT_ROSTRO, name=f"rostro_dropout{i}")(x)
    salida = layers.Dense(1, activation="sigmoid", name="es_mi_rostro")(x)
    return keras.Model(entrada, salida, name="desbloqueo_facial")


def congelar(base):
    """Congela TODA la base: sus pesos no se actualizan al entrenar.

    Por que: el clasificador nuevo empieza con pesos aleatorios, asi que en
    las primeras epocas sus gradientes son grandes y "ruidosos". Si la base
    estuviera descongelada, esos gradientes se propagarian hacia atras y
    podrian "arruinar" las caracteristicas aprendidas con CelebA. Primero se
    entrena solo el clasificador y despues (paso 3) se refina todo con cuidado.

    Nota: hay que congelar ANTES de compilar el modelo (model.compile), porque
    Keras decide que pesos optimizar en el momento de compilar.
    """
    base.trainable = False


def descongelar_ultimos_bloques(base, n_bloques):
    """Descongela solo los ultimos `n_bloques` bloques convolucionales.

    Los primeros bloques detectan rasgos muy genericos (bordes, colores) que
    sirven igual para cualquier cara, asi que se dejan congelados. Los ultimos
    bloques codifican rasgos mas especificos; ajustarlos un poco ayuda a
    distinguir MI cara.

    Las capas BatchNormalization se dejan congeladas: con lotes pequenos y
    pocos datos, actualizar sus medias/varianzas moveria las estadisticas
    aprendidas con CelebA y degradaria la red. En Keras, una BN con
    trainable=False ademas funciona en modo inferencia.
    """
    n_total = len(config.FILTROS)
    bloques = {f"bloque{i}" for i in range(n_total - n_bloques + 1, n_total + 1)}
    base.trainable = True
    for capa in base.layers:
        prefijo = capa.name.split("_")[0]
        es_bn = isinstance(capa, layers.BatchNormalization)
        capa.trainable = (prefijo in bloques) and not es_bn
    return sorted(bloques)


def contar_parametros(modelo):
    """(entrenables, congelados) para verificar que el congelado funciono."""
    entrenables = sum(int(w.numpy().size) for w in modelo.trainable_weights)
    congelados = sum(int(w.numpy().size) for w in modelo.non_trainable_weights)
    return entrenables, congelados


def compilar_binario(modelo, lr):
    """Compila el clasificador binario con Adam y metricas utiles."""
    modelo.compile(
        optimizer=keras.optimizers.Adam(lr),
        loss="binary_crossentropy",
        metrics=[
            keras.metrics.BinaryAccuracy(name="accuracy"),
            keras.metrics.AUC(name="auc"),
            keras.metrics.Precision(name="precision"),
            keras.metrics.Recall(name="recall"),
        ],
    )
