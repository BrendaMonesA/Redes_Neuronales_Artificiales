"""
Problema 2: Aproximar funciones con una red neuronal en [-1, 1].

    (a) f(x) = 3 sin(pi x)
    (b) f(x) = 1 + 2x + 4x^3

Se entrena una MLP para cada funcion y se grafica la salida de la red junto
con la funcion real.
"""

import os
import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FIG_DIR = os.path.join(os.path.dirname(__file__), "figuras")
os.makedirs(FIG_DIR, exist_ok=True)

tf.random.set_seed(42)
np.random.seed(42)


def construir_red():
    """MLP pequena: 1 -> 64 -> 64 -> 1 con activacion tanh."""
    return tf.keras.Sequential([
        tf.keras.layers.Input(shape=(1,)),
        tf.keras.layers.Dense(64, activation="tanh"),
        tf.keras.layers.Dense(64, activation="tanh"),
        tf.keras.layers.Dense(1),
    ])


def entrenar_y_graficar(func, nombre, titulo, archivo, epochs=2000):
    print("=" * 60)
    print(f"Ajustando {nombre}: {titulo}")
    print("=" * 60)

    x_train = np.linspace(-1, 1, 400).reshape(-1, 1).astype(np.float32)
    y_train = func(x_train).astype(np.float32)

    modelo = construir_red()
    modelo.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="mse")
    hist = modelo.fit(x_train, y_train, epochs=epochs, batch_size=64,
                      verbose=0)
    print(f"MSE final: {hist.history['loss'][-1]:.3e}")

    x_test = np.linspace(-1, 1, 1000).reshape(-1, 1).astype(np.float32)
    y_true = func(x_test)
    y_pred = modelo.predict(x_test, verbose=0)

    plt.figure(figsize=(8, 5))
    plt.plot(x_test, y_true, "b-", lw=2, label="Funcion real")
    plt.plot(x_test, y_pred, "r--", lw=2, label="Red neuronal")
    plt.title(f"{nombre}: {titulo}")
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    out = os.path.join(FIG_DIR, archivo)
    plt.tight_layout()
    plt.savefig(out, dpi=120)
    plt.close()
    print(f"Figura guardada en: {out}")


def main():
    entrenar_y_graficar(
        lambda x: 3 * np.sin(np.pi * x),
        "(a)", "3 sin(pi x)", "p2a_3sin_pix.png",
    )
    entrenar_y_graficar(
        lambda x: 1 + 2 * x + 4 * x ** 3,
        "(b)", "1 + 2x + 4x^3", "p2b_polinomio.png",
    )


if __name__ == "__main__":
    main()
