"""
Problema 3: Capa entrenable que representa un polinomio de grado 3.

    f(x) = a0 + a1 x + a2 x^2 + a3 x^3

Los parametros entrenables son a0, a1, a2, a3. Se entrena la capa para ajustar
f(x) = cos(2x) en el intervalo [-1, 1].
"""

import os
import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FIG_DIR = os.path.join(os.path.dirname(__file__), "figuras")
os.makedirs(FIG_DIR, exist_ok=True)

tf.random.set_seed(0)
np.random.seed(0)


class CapaPolinomio3(tf.keras.layers.Layer):
    """Capa entrenable: f(x) = a0 + a1 x + a2 x^2 + a3 x^3.

    Entrada (N, 1) -> salida (N, 1). Los 4 coeficientes son pesos entrenables.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, input_shape):
        self.coefs = self.add_weight(
            name="coeficientes",
            shape=(4,),
            initializer="zeros",
            trainable=True,
        )
        super().build(input_shape)

    def call(self, inputs):
        x = tf.cast(inputs, tf.float32)
        # Potencias x^0, x^1, x^2, x^3 -> (N, 4)
        potencias = tf.concat([x ** k for k in range(4)], axis=-1)
        # Producto por los coeficientes -> (N, 1)
        return tf.reduce_sum(potencias * self.coefs, axis=-1, keepdims=True)


def main():
    print("=" * 60)
    print("Problema 3: capa polinomial grado 3 ajustando cos(2x)")
    print("=" * 60)

    x_train = np.linspace(-1, 1, 400).reshape(-1, 1).astype(np.float32)
    y_train = np.cos(2 * x_train).astype(np.float32)

    modelo = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(1,)),
        CapaPolinomio3(),
    ])
    modelo.compile(optimizer=tf.keras.optimizers.Adam(5e-2), loss="mse")
    hist = modelo.fit(x_train, y_train, epochs=3000, batch_size=64, verbose=0)
    print(f"MSE final: {hist.history['loss'][-1]:.3e}")

    coefs = modelo.layers[0].coefs.numpy()
    print(f"Coeficientes aprendidos: a0={coefs[0]:.4f}, a1={coefs[1]:.4f}, "
          f"a2={coefs[2]:.4f}, a3={coefs[3]:.4f}")
    # Referencia: serie de Taylor de cos(2x) ~ 1 - 2x^2 (+ 0 x, 0 x^3)
    print("Referencia Taylor cos(2x) ~ 1 + 0x - 2x^2 + 0x^3")

    x_test = np.linspace(-1, 1, 1000).reshape(-1, 1).astype(np.float32)
    y_true = np.cos(2 * x_test)
    y_pred = modelo.predict(x_test, verbose=0)

    plt.figure(figsize=(8, 5))
    plt.plot(x_test, y_true, "b-", lw=2, label="cos(2x)")
    plt.plot(x_test, y_pred, "r--", lw=2,
             label="Polinomio grado 3 (capa entrenable)")
    plt.title("Capa polinomial grado 3 ajustando cos(2x) en [-1, 1]")
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    txt = (f"a0={coefs[0]:.3f}\na1={coefs[1]:.3f}\n"
           f"a2={coefs[2]:.3f}\na3={coefs[3]:.3f}")
    plt.gca().text(0.02, 0.02, txt, transform=plt.gca().transAxes,
                   va="bottom", ha="left", fontsize=9,
                   bbox=dict(boxstyle="round", fc="w", alpha=0.8))
    out = os.path.join(FIG_DIR, "p3_polinomio_cos2x.png")
    plt.tight_layout()
    plt.savefig(out, dpi=120)
    plt.close()
    print(f"Figura guardada en: {out}")


if __name__ == "__main__":
    main()
