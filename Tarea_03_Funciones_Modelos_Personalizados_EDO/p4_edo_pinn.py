"""
Problema 4: Resolver ecuaciones diferenciales con una red neuronal (PINN)
en el intervalo [-5, 5]. Se grafica la solucion de la red junto con la
solucion analitica.

    (a) x y' + y = x^2 cos(x),   y(0) = 0
        Nota: x y' + y = (x y)', por lo que  x y = integral(x^2 cos x) dx.
        Solucion analitica:  y(x) = x sin(x) + 2 cos(x) - 2 sin(x)/x
        (en x=0 el limite vale y(0)=0).

    (b) y'' = -y,   y(0) = 1,  y'(0) = -0.5
        Solucion analitica:  y(x) = cos(x) - 0.5 sin(x)

Se entrena minimizando el residuo de la EDO mas las condiciones iniciales.
Las derivadas se calculan con diferenciacion automatica (tf.GradientTape).
"""

import os
import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FIG_DIR = os.path.join(os.path.dirname(__file__), "figuras")
os.makedirs(FIG_DIR, exist_ok=True)

tf.random.set_seed(7)
np.random.seed(7)

A, B = -5.0, 5.0


def construir_red():
    return tf.keras.Sequential([
        tf.keras.layers.Input(shape=(1,)),
        tf.keras.layers.Dense(64, activation="tanh"),
        tf.keras.layers.Dense(64, activation="tanh"),
        tf.keras.layers.Dense(64, activation="tanh"),
        tf.keras.layers.Dense(1),
    ])


# ---------------------------------------------------------------------------
# (a) x y' + y = x^2 cos(x), y(0)=0
# ---------------------------------------------------------------------------
def sol_analitica_a(x):
    x = np.asarray(x, dtype=np.float64)
    y = np.empty_like(x)
    chico = np.abs(x) < 1e-6
    # Limite en x=0: y -> 0
    y[chico] = 0.0
    xn = x[~chico]
    y[~chico] = xn * np.sin(xn) + 2 * np.cos(xn) - 2 * np.sin(xn) / xn
    return y


def resolver_a(epochs=8000, n_col=400):
    print("=" * 60)
    print("(a) x y' + y = x^2 cos(x),  y(0)=0")
    print("=" * 60)
    modelo = construir_red()
    opt = tf.keras.optimizers.Adam(1e-3)
    x_col = tf.constant(np.linspace(A, B, n_col).reshape(-1, 1), dtype=tf.float32)
    x0 = tf.constant([[0.0]], dtype=tf.float32)

    @tf.function
    def paso():
        with tf.GradientTape() as tape_w:
            with tf.GradientTape() as tape_x:
                tape_x.watch(x_col)
                y = modelo(x_col)
            dy = tape_x.gradient(y, x_col)
            residuo = x_col * dy + y - x_col ** 2 * tf.cos(x_col)
            perdida_edo = tf.reduce_mean(residuo ** 2)
            y0 = modelo(x0)
            perdida_ic = tf.reduce_mean((y0 - 0.0) ** 2)
            perdida = perdida_edo + 10.0 * perdida_ic
        grads = tape_w.gradient(perdida, modelo.trainable_variables)
        opt.apply_gradients(zip(grads, modelo.trainable_variables))
        return perdida

    for e in range(epochs):
        p = paso()
        if (e + 1) % 2000 == 0:
            print(f"  epoch {e+1:5d}  perdida={p.numpy():.3e}")
    return modelo


# ---------------------------------------------------------------------------
# (b) y'' = -y, y(0)=1, y'(0)=-0.5
# ---------------------------------------------------------------------------
def sol_analitica_b(x):
    x = np.asarray(x, dtype=np.float64)
    return np.cos(x) - 0.5 * np.sin(x)


def resolver_b(epochs=8000, n_col=400):
    print("=" * 60)
    print("(b) y'' = -y,  y(0)=1, y'(0)=-0.5")
    print("=" * 60)
    modelo = construir_red()
    opt = tf.keras.optimizers.Adam(1e-3)
    x_col = tf.constant(np.linspace(A, B, n_col).reshape(-1, 1), dtype=tf.float32)
    x0 = tf.constant([[0.0]], dtype=tf.float32)

    @tf.function
    def paso():
        with tf.GradientTape() as tape_w:
            with tf.GradientTape() as t2:
                t2.watch(x_col)
                with tf.GradientTape() as t1:
                    t1.watch(x_col)
                    y = modelo(x_col)
                dy = t1.gradient(y, x_col)
            d2y = t2.gradient(dy, x_col)
            residuo = d2y + y
            perdida_edo = tf.reduce_mean(residuo ** 2)
            # Condiciones iniciales: y(0)=1, y'(0)=-0.5
            with tf.GradientTape() as t0:
                t0.watch(x0)
                y0 = modelo(x0)
            dy0 = t0.gradient(y0, x0)
            perdida_ic = tf.reduce_mean((y0 - 1.0) ** 2) + \
                tf.reduce_mean((dy0 - (-0.5)) ** 2)
            perdida = perdida_edo + 10.0 * perdida_ic
        grads = tape_w.gradient(perdida, modelo.trainable_variables)
        opt.apply_gradients(zip(grads, modelo.trainable_variables))
        return perdida

    for e in range(epochs):
        p = paso()
        if (e + 1) % 2000 == 0:
            print(f"  epoch {e+1:5d}  perdida={p.numpy():.3e}")
    return modelo


def graficar(modelo, sol_analitica, titulo, archivo):
    x_test = np.linspace(A, B, 1000).reshape(-1, 1).astype(np.float32)
    y_pred = modelo.predict(x_test, verbose=0).flatten()
    y_true = sol_analitica(x_test.flatten())
    err = np.sqrt(np.mean((y_pred - y_true) ** 2))
    print(f"  RMSE red vs analitica: {err:.3e}")

    plt.figure(figsize=(8, 5))
    plt.plot(x_test, y_true, "b-", lw=2, label="Solucion analitica")
    plt.plot(x_test, y_pred, "r--", lw=2, label="Red neuronal (PINN)")
    plt.title(titulo)
    plt.xlabel("x")
    plt.ylabel("y(x)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    out = os.path.join(FIG_DIR, archivo)
    plt.tight_layout()
    plt.savefig(out, dpi=120)
    plt.close()
    print(f"  Figura guardada en: {out}")


def main():
    m_a = resolver_a()
    graficar(m_a, sol_analitica_a,
             "(a)  x y' + y = x^2 cos(x),  y(0)=0",
             "p4a_edo_primer_orden.png")

    m_b = resolver_b()
    graficar(m_b, sol_analitica_b,
             "(b)  y'' = -y,  y(0)=1, y'(0)=-0.5",
             "p4b_edo_segundo_orden.png")


if __name__ == "__main__":
    main()
