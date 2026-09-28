"""
Metricas y graficas para evaluar el desbloqueo facial.

Para un sistema de desbloqueo la exactitud (accuracy) sola no basta. Se usan
las metricas clasicas de biometria:

* TAR / recall (True Acceptance Rate): de las veces que YO intento
  desbloquear, cuantas veces me deja entrar.
* FRR (False Rejection Rate) = 1 - TAR: veces que me rechaza a mi.
  Molesto, pero no peligroso.
* FAR (False Acceptance Rate): de las veces que OTRA persona lo intenta,
  cuantas veces la deja entrar. Este es el error peligroso.

El umbral de decision (por defecto 0.5) intercambia FAR por FRR: subirlo
hace al sistema mas estricto (menos FAR, mas FRR).
"""

import json

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc as area_bajo_curva

import config


def predecir(modelo, ds):
    """Recorre un dataset y regresa (etiquetas reales, probabilidades)."""
    y_real, y_prob = [], []
    for x, y in ds:
        y_real.append(y.numpy().ravel())
        y_prob.append(modelo.predict_on_batch(x).ravel())
    return np.concatenate(y_real), np.concatenate(y_prob)


def metricas_binarias(y_real, puntaje, umbral=0.5):
    """Matriz de confusion y metricas biometricas para un umbral dado.

    `puntaje` puede ser una probabilidad (red) o una distancia al hiperplano
    (SVM, con umbral 0). Se predice "es mi rostro" si puntaje >= umbral.
    """
    y_real = np.asarray(y_real).astype(int)
    pred = (np.asarray(puntaje) >= umbral).astype(int)
    tp = int(np.sum((pred == 1) & (y_real == 1)))
    tn = int(np.sum((pred == 0) & (y_real == 0)))
    fp = int(np.sum((pred == 1) & (y_real == 0)))
    fn = int(np.sum((pred == 0) & (y_real == 1)))
    tar = tp / max(tp + fn, 1)
    far = fp / max(fp + tn, 1)
    return {
        "umbral": float(umbral),
        "TP": tp, "TN": tn, "FP": fp, "FN": fn,
        "accuracy": (tp + tn) / max(len(y_real), 1),
        # Promedio de la exactitud en cada clase: no se infla si hay muchos
        # mas negativos que positivos (como en validacion).
        "accuracy_balanceada": (tar + (1 - far)) / 2,
        "precision": tp / max(tp + fp, 1),
        "TAR_recall": tar,
        "FRR": 1 - tar,
        "FAR": far,
    }


def umbral_para_far(y_real, puntaje, far_max=config.FAR_OBJETIVO):
    """Umbral mas bajo (el que mas me acepta a mi) con FAR <= far_max.

    Se prueban como candidatos todos los puntajes observados; para cada uno
    se calcula el FAR y se elige el menor que cumpla el objetivo.
    """
    y_real = np.asarray(y_real).astype(int)
    puntaje = np.asarray(puntaje)
    candidatos = np.unique(puntaje)
    negativos = puntaje[y_real == 0]
    for u in candidatos:
        if np.mean(negativos >= u) <= far_max:
            return float(u)
    return float(candidatos[-1])


def imprimir_metricas(titulo, m):
    print(f"\n--- {titulo} (umbral = {m['umbral']:.4f}) ---")
    print(f"  Matriz de confusion: TP={m['TP']}  FN={m['FN']}  "
          f"FP={m['FP']}  TN={m['TN']}")
    print(f"  Accuracy            : {m['accuracy']:.4f}")
    print(f"  Accuracy balanceada : {m['accuracy_balanceada']:.4f}")
    print(f"  Precision           : {m['precision']:.4f}")
    print(f"  TAR (me acepta)     : {m['TAR_recall']:.4f}")
    print(f"  FRR (me rechaza)    : {m['FRR']:.4f}")
    print(f"  FAR (acepta a otro) : {m['FAR']:.4f}")


def guardar_umbral(umbral, metricas, ruta=config.UMBRAL_JSON):
    with open(ruta, "w", encoding="utf-8") as f:
        json.dump({"umbral": umbral, "far_objetivo": config.FAR_OBJETIVO,
                   "metricas_validacion": metricas}, f, indent=2)


def cargar_umbral(ruta=config.UMBRAL_JSON):
    try:
        with open(ruta, encoding="utf-8") as f:
            return float(json.load(f)["umbral"])
    except (OSError, KeyError, ValueError):
        return 0.5


# ===========================================================================
# Graficas
# ===========================================================================
def graficar_historial(historial, ruta, metricas=("loss", "accuracy"),
                       titulo=""):
    """Curvas de entrenamiento vs. validacion por epoca.

    Si la curva de validacion se separa hacia arriba de la de entrenamiento
    (en la perdida) hay sobreajuste.
    """
    h = historial.history if hasattr(historial, "history") else historial
    metricas = [m for m in metricas if m in h]
    fig, ejes = plt.subplots(1, len(metricas), figsize=(6 * len(metricas), 4))
    ejes = np.atleast_1d(ejes)
    for eje, m in zip(ejes, metricas):
        eje.plot(h[m], label="entrenamiento")
        if f"val_{m}" in h:
            eje.plot(h[f"val_{m}"], label="validacion")
        eje.set_xlabel("epoca")
        eje.set_ylabel(m)
        eje.grid(alpha=0.3)
        eje.legend()
    fig.suptitle(titulo)
    fig.tight_layout()
    fig.savefig(ruta, dpi=120)
    plt.close(fig)


def graficar_roc(y_real, puntaje, ruta, titulo=""):
    """Curva ROC: TAR (eje y) contra FAR (eje x) para todos los umbrales."""
    far, tar, _ = roc_curve(y_real, puntaje)
    fig, eje = plt.subplots(figsize=(5, 5))
    eje.plot(far, tar, lw=2, label=f"AUC = {area_bajo_curva(far, tar):.4f}")
    eje.plot([0, 1], [0, 1], "k--", lw=1, label="azar")
    eje.set_xlabel("FAR (acepta a otra persona)")
    eje.set_ylabel("TAR (me acepta a mi)")
    eje.set_title(titulo)
    eje.grid(alpha=0.3)
    eje.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(ruta, dpi=120)
    plt.close(fig)


def graficar_matriz_confusion(m, ruta, titulo=""):
    matriz = np.array([[m["TN"], m["FP"]], [m["FN"], m["TP"]]])
    fig, eje = plt.subplots(figsize=(4.5, 4))
    eje.imshow(matriz, cmap="Blues")
    for i in range(2):
        for j in range(2):
            eje.text(j, i, matriz[i, j], ha="center", va="center",
                     color="white" if matriz[i, j] > matriz.max() / 2
                     else "black", fontsize=13)
    eje.set_xticks([0, 1], ["pred: otro", "pred: yo"])
    eje.set_yticks([0, 1], ["real: otro", "real: yo"])
    eje.set_title(f"{titulo}\numbral = {m['umbral']:.3f}")
    fig.tight_layout()
    fig.savefig(ruta, dpi=120)
    plt.close(fig)


def graficar_histograma_puntajes(y_real, puntaje, umbral, ruta, titulo=""):
    """Distribucion de probabilidades para mis fotos y las de otros.

    Un buen modelo separa las dos montanas; el umbral es la linea vertical.
    """
    y_real = np.asarray(y_real).astype(int)
    fig, eje = plt.subplots(figsize=(7, 4))
    bins = np.linspace(min(0, puntaje.min()), max(1, puntaje.max()), 41)
    eje.hist(puntaje[y_real == 0], bins=bins, alpha=0.6, label="otros",
             density=True)
    eje.hist(puntaje[y_real == 1], bins=bins, alpha=0.6, label="yo",
             density=True)
    eje.axvline(umbral, color="k", ls="--", label=f"umbral {umbral:.3f}")
    eje.set_xlabel("puntaje / probabilidad de ser yo")
    eje.set_ylabel("densidad")
    eje.set_title(titulo)
    eje.legend()
    fig.tight_layout()
    fig.savefig(ruta, dpi=120)
    plt.close(fig)


def evaluar_y_graficar(modelo, ds_val, prefijo, titulo):
    """Evaluacion completa del clasificador sobre validacion.

    Imprime metricas con umbral 0.5 y con el umbral que cumple FAR_OBJETIVO,
    guarda ROC, matriz de confusion e histograma. Regresa (umbral, metricas).
    """
    y, p = predecir(modelo, ds_val)
    m05 = metricas_binarias(y, p, 0.5)
    imprimir_metricas(f"{titulo} - umbral por defecto", m05)

    u = umbral_para_far(y, p)
    m_u = metricas_binarias(y, p, u)
    imprimir_metricas(f"{titulo} - umbral para FAR <= {config.FAR_OBJETIVO}",
                      m_u)

    graficar_roc(y, p, config.FIG_DIR / f"{prefijo}_roc.png", titulo)
    graficar_matriz_confusion(m_u, config.FIG_DIR / f"{prefijo}_confusion.png",
                              titulo)
    graficar_histograma_puntajes(y, p, u,
                                 config.FIG_DIR / f"{prefijo}_puntajes.png",
                                 titulo)
    return u, {"umbral_0.5": m05, "umbral_far": m_u}
