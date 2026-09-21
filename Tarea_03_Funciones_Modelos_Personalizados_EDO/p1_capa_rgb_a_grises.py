"""
Problema 1: Capa de Keras que transforma imagenes RGB a escala de grises.

Solo se disena la capa: no hay parametros que entrenar. Se usa la conversion
estandar de luminosidad (ITU-R BT.601):

    gris = 0.299 R + 0.587 G + 0.114 B

La capa acepta un batch de imagenes (H, W, 3) y devuelve (H, W, 1).
Se prueba sobre imagenes RGB sinteticas (no requiere descargar datos).
"""

import os
import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FIG_DIR = os.path.join(os.path.dirname(__file__), "figuras")
os.makedirs(FIG_DIR, exist_ok=True)


class RGBaGrises(tf.keras.layers.Layer):
    """Convierte imagenes RGB (..., 3) a escala de grises (..., 1).

    No tiene pesos entrenables. Los coeficientes de luminosidad se guardan
    como una constante no entrenable.
    """

    def __init__(self, pesos=(0.299, 0.587, 0.114), **kwargs):
        super().__init__(**kwargs)
        self.pesos = pesos

    def build(self, input_shape):
        if input_shape[-1] != 3:
            raise ValueError(
                f"Se esperaban 3 canales (RGB), se recibieron {input_shape[-1]}."
            )
        # Constante (no entrenable) con forma (3, 1) para hacer el producto por canal.
        self.kernel = tf.constant(
            np.array(self.pesos, dtype=np.float32).reshape(3, 1)
        )
        super().build(input_shape)

    def call(self, inputs):
        inputs = tf.cast(inputs, tf.float32)
        # (..., 3) x (3, 1) -> (..., 1)
        return tf.matmul(inputs, self.kernel)

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"pesos": self.pesos})
        return cfg


def imagenes_rgb_sinteticas(n=3, size=64):
    """Genera n imagenes RGB sinteticas con formas de colores en [0, 1]."""
    rng = np.random.default_rng(0)
    imgs = np.zeros((n, size, size, 3), dtype=np.float32)
    yy, xx = np.mgrid[0:size, 0:size] / size
    for i in range(n):
        # Fondo: gradiente de color
        imgs[i, ..., 0] = xx
        imgs[i, ..., 1] = yy
        imgs[i, ..., 2] = (1 - xx) * (1 - yy)
        # Circulo de color aleatorio
        cx, cy = rng.uniform(0.2, 0.8, size=2)
        r = rng.uniform(0.1, 0.25)
        mask = (xx - cx) ** 2 + (yy - cy) ** 2 < r ** 2
        color = rng.uniform(0, 1, size=3)
        for c in range(3):
            imgs[i, ..., c][mask] = color[c]
    return imgs


def main():
    print("=" * 60)
    print("Problema 1: capa RGB -> escala de grises")
    print("=" * 60)

    # Modelo minimo que solo aplica la capa.
    entrada = tf.keras.Input(shape=(None, None, 3))
    salida = RGBaGrises()(entrada)
    modelo = tf.keras.Model(entrada, salida, name="rgb_a_grises")
    modelo.summary()

    imgs = imagenes_rgb_sinteticas(n=3, size=64)
    grises = modelo.predict(imgs, verbose=0)
    print(f"Entrada: {imgs.shape}  ->  Salida: {grises.shape}")

    # Figura comparativa
    n = imgs.shape[0]
    fig, axes = plt.subplots(2, n, figsize=(3 * n, 6))
    for i in range(n):
        axes[0, i].imshow(imgs[i])
        axes[0, i].set_title(f"RGB #{i}")
        axes[0, i].axis("off")
        axes[1, i].imshow(grises[i, ..., 0], cmap="gray")
        axes[1, i].set_title(f"Grises #{i}")
        axes[1, i].axis("off")
    fig.suptitle("Capa RGBaGrises  (0.299R + 0.587G + 0.114B)")
    fig.tight_layout()
    out = os.path.join(FIG_DIR, "p1_rgb_a_grises.png")
    fig.savefig(out, dpi=120)
    print(f"Figura guardada en: {out}")

    # Verificacion numerica contra la formula directa
    ref = (0.299 * imgs[..., 0] + 0.587 * imgs[..., 1] + 0.114 * imgs[..., 2])
    err = np.max(np.abs(ref - grises[..., 0]))
    print(f"Error maximo vs formula directa: {err:.2e}")


if __name__ == "__main__":
    main()
