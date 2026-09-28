"""
Paso 1 (estrategia 2.1.1 y 2.2.3a): pre-entrenar una CNN con CelebA.

Objetivo: que la red aprenda a "entender" rostros. Para eso la entrenamos a
predecir los 40 atributos de CelebA (sonrie, lentes, barba, cabello rubio,
hombre, joven, ...). Para acertar esos atributos la red tiene que aprender
rasgos faciales generales: forma de ojos, boca, cejas, contorno, cabello...
Justo esos rasgos son los que despues sirven para reconocer MI rostro, aunque
CelebA no tenga ninguna foto mia.

Esto es lo que el articulo de referencia (Razavian et al., 2014) llama usar
una red como extractor de caracteristicas "off-the-shelf": una red entrenada
para una tarea (alli ImageNet, aqui atributos de CelebA) produce una
representacion que funciona muy bien para otras tareas relacionadas.

Salida:
    modelos/cnn_celeba.keras        modelo completo (base + clasificador)
    modelos/atributos_celeba.json   nombres de los 40 atributos
    figuras/p1_historial.png        curvas de entrenamiento
    figuras/p1_accuracy_atributos.png  exactitud por atributo en prueba

Uso:
    python paso1_preentrenar_celeba.py
    python paso1_preentrenar_celeba.py --epocas 5 --max-train 20000
"""

import argparse
import json

import numpy as np
import tensorflow as tf
import keras
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config
import datos
import modelos
import evaluacion


def argumentos():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("--epocas", type=int, default=config.EPOCAS_CELEBA)
    p.add_argument("--max-train", type=int, default=config.CELEBA_MAX_TRAIN,
                   help="imagenes de entrenamiento (0 = todas)")
    p.add_argument("--max-val", type=int, default=config.CELEBA_MAX_VAL)
    p.add_argument("--max-test", type=int, default=config.CELEBA_MAX_TEST)
    return p.parse_args()


def accuracy_por_atributo(modelo, ds, nombres):
    """Exactitud de cada uno de los 40 atributos sobre un dataset.

    Referencia util: la "linea base" de cada atributo es la frecuencia de la
    clase mayoritaria (ej. solo ~6% tiene lentes, asi que decir siempre "no
    tiene lentes" ya da 94%). La red debe superar esa linea base.
    """
    aciertos = np.zeros(len(nombres))
    positivos = np.zeros(len(nombres))
    n = 0
    for x, y in ds:
        prob = modelo.predict_on_batch(x)
        y = y.numpy()
        aciertos += np.sum((prob >= 0.5) == (y >= 0.5), axis=0)
        positivos += np.sum(y, axis=0)
        n += len(y)
    acc = aciertos / n
    frecuencia = positivos / n
    linea_base = np.maximum(frecuencia, 1 - frecuencia)
    return acc, linea_base


def graficar_atributos(nombres, acc, linea_base, ruta):
    orden = np.argsort(acc)
    fig, eje = plt.subplots(figsize=(8, 11))
    y = np.arange(len(nombres))
    eje.barh(y, acc[orden], color="tab:blue", label="CNN")
    eje.scatter(linea_base[orden], y, color="tab:red", marker="|", s=120,
                label="linea base (clase mayoritaria)", zorder=3)
    eje.set_yticks(y, [nombres[i] for i in orden], fontsize=8)
    eje.set_xlim(0.5, 1.0)
    eje.set_xlabel("accuracy en prueba")
    eje.set_title(f"CelebA: accuracy por atributo (media = {acc.mean():.3f})")
    eje.legend(loc="lower right")
    eje.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(ruta, dpi=120)
    plt.close(fig)


def main():
    args = argumentos()
    config.crear_carpetas_salida()
    keras.utils.set_random_seed(config.SEMILLA)

    print("=" * 70)
    print("Paso 1: pre-entrenamiento de la CNN con CelebA (40 atributos)")
    print("=" * 70)

    # -------------------------------------------------------------- datos
    ds_train, nombres, n_train = datos.dataset_celeba(
        0, args.max_train or None, entrenamiento=True)
    ds_val, _, n_val = datos.dataset_celeba(1, args.max_val or None)
    ds_test, _, n_test = datos.dataset_celeba(2, args.max_test or None)
    print(f"Imagenes: train={n_train}  val={n_val}  test={n_test}")
    print(f"Atributos ({len(nombres)}): {', '.join(nombres)}")
    with open(config.ATRIBUTOS_JSON, "w", encoding="utf-8") as f:
        json.dump(nombres, f, indent=2)

    # -------------------------------------------------------------- modelo
    modelo = modelos.construir_cnn_celeba(len(nombres))
    modelo.summary()

    # Perdida: entropia cruzada binaria (una por atributo, promediada).
    # Metricas: accuracy binaria promedio y AUC multi-etiqueta (promedio del
    # AUC de cada atributo; no depende del umbral ni del desbalance).
    modelo.compile(
        optimizer=keras.optimizers.Adam(config.LR_CELEBA),
        loss="binary_crossentropy",
        metrics=[keras.metrics.BinaryAccuracy(name="accuracy"),
                 keras.metrics.AUC(name="auc", multi_label=True,
                                   num_labels=len(nombres))],
    )

    # Callbacks:
    #  * ModelCheckpoint guarda el mejor modelo segun la perdida de validacion
    #  * ReduceLROnPlateau baja el learning rate si la validacion se estanca
    #  * EarlyStopping detiene el entrenamiento si deja de mejorar
    callbacks = [
        keras.callbacks.ModelCheckpoint(str(config.MODELO_CELEBA),
                                        monitor="val_loss",
                                        save_best_only=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                                          patience=2, verbose=1),
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=4,
                                      restore_best_weights=True, verbose=1),
    ]

    historial = modelo.fit(ds_train, validation_data=ds_val,
                           epochs=args.epocas, callbacks=callbacks)
    modelo.save(config.MODELO_CELEBA)
    print(f"\nModelo guardado en {config.MODELO_CELEBA}")

    evaluacion.graficar_historial(historial, config.FIG_DIR / "p1_historial.png",
                                  ("loss", "accuracy", "auc"),
                                  "Paso 1: CNN en CelebA (40 atributos)")

    # ---------------------------------------------------------- evaluacion
    resultado = modelo.evaluate(ds_test, return_dict=True, verbose=0)
    print(f"\nPrueba: loss={resultado['loss']:.4f}  "
          f"accuracy={resultado['accuracy']:.4f}  auc={resultado['auc']:.4f}")

    acc, linea_base = accuracy_por_atributo(modelo, ds_test, nombres)
    print("\nAccuracy por atributo (prueba) vs. linea base:")
    for nombre, a, b in sorted(zip(nombres, acc, linea_base),
                               key=lambda t: t[1] - t[2], reverse=True):
        print(f"  {nombre:22s} {a:.3f}  (base {b:.3f}, mejora {a - b:+.3f})")
    graficar_atributos(nombres, acc, linea_base,
                       config.FIG_DIR / "p1_accuracy_atributos.png")
    print(f"\nFiguras en {config.FIG_DIR}")


if __name__ == "__main__":
    tf.get_logger().setLevel("ERROR")
    main()
