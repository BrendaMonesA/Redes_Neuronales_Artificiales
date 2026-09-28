"""
Paso 2 (estrategia 2.1.2 - 2.1.5 y 2.2.3b): transfer learning con la base
congelada.

1. Se carga la CNN entrenada en CelebA (paso 1).
2. Se le QUITA el clasificador (las capas densas del final: las que
   predecian los 40 atributos) y nos quedamos solo con la base convolucional.
3. Se le AGREGA un clasificador nuevo: 1 o 2 capas densas + una sola
   neurona de salida con sigmoide -> P(la foto es mia).
4. Se CONGELAN los pesos de la base pre-entrenada.
5. Se entrena el modelo: solo cambian las capas densas nuevas.

Como la base ya sabe extraer rasgos faciales, el clasificador nuevo solo
tiene que aprender "que combinacion de rasgos es la mia", lo cual se puede
lograr con pocas fotos. Para compensar que tenemos pocas fotos propias se
usa aumento de datos sintetico (ver datos.construir_aumento) y se equilibra
el conjunto con el mismo numero de rostros de otras personas.

Salida:
    modelos/rostro_etapa2.keras
    modelos/umbral.json               (se sobrescribe en el paso 3)
    figuras/p2_ejemplos_aumento.png   como se ven las fotos aumentadas
    figuras/p2_historial.png, p2_roc.png, p2_confusion.png, p2_puntajes.png

Uso:
    python paso2_entrenar_clasificador.py
"""

import argparse

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
    p = argparse.ArgumentParser(description="Paso 2: clasificador con base "
                                "congelada")
    p.add_argument("--epocas", type=int, default=config.EPOCAS_ETAPA2)
    return p.parse_args()


def graficar_ejemplos_aumento(ds_train, ruta, n=12):
    """Muestra un lote de entrenamiento ya aumentado (con su etiqueta)."""
    x, y = next(iter(ds_train))
    n = min(n, len(x))
    columnas = 6
    filas = int(np.ceil(n / columnas))
    fig, ejes = plt.subplots(filas, columnas, figsize=(2 * columnas, 2.2 * filas))
    for i, eje in enumerate(np.ravel(ejes)):
        eje.axis("off")
        if i < n:
            eje.imshow(x[i].numpy())
            eje.set_title("yo" if y[i, 0] > 0.5 else "otro", fontsize=9)
    fig.suptitle("Lote de entrenamiento con aumento de datos")
    fig.tight_layout()
    fig.savefig(ruta, dpi=110)
    plt.close(fig)


def main():
    args = argumentos()
    config.crear_carpetas_salida()
    keras.utils.set_random_seed(config.SEMILLA)

    print("=" * 70)
    print("Paso 2: base pre-entrenada CONGELADA + clasificador nuevo")
    print("=" * 70)

    # -------------------------------------------------------------- datos
    ds_train, ds_val, resumen = datos.datasets_mi_rostro()
    for clave, valor in resumen.items():
        print(f"  {clave:18s}: {valor}")
    graficar_ejemplos_aumento(ds_train,
                              config.FIG_DIR / "p2_ejemplos_aumento.png")

    # ------------------------------------------ 1-2: cargar y quitar cabeza
    if not config.MODELO_CELEBA.exists():
        raise FileNotFoundError(f"No existe {config.MODELO_CELEBA}. "
                                "Ejecuta primero paso1_preentrenar_celeba.py")
    cnn_celeba = keras.models.load_model(config.MODELO_CELEBA)
    base = modelos.extraer_base(cnn_celeba)
    print(f"\nBase extraida de {config.MODELO_CELEBA.name}: "
          f"salida {base.output.shape}")

    # ------------------------------------ 3: nuevo clasificador de 1 neurona
    modelo = modelos.construir_clasificador_rostro(base)

    # -------------------------------------------- 4: congelar (antes de compile)
    modelos.congelar(base)
    entrenables, congelados = modelos.contar_parametros(modelo)
    print(f"Parametros entrenables: {entrenables:,}  congelados: {congelados:,}")
    modelo.summary()

    # ---------------------------------------------------------- 5: entrenar
    modelos.compilar_binario(modelo, config.LR_ETAPA2)
    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=5,
                                      restore_best_weights=True, verbose=1),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                                          patience=2, verbose=1),
    ]
    historial = modelo.fit(ds_train, validation_data=ds_val,
                           epochs=args.epocas, callbacks=callbacks)
    modelo.save(config.MODELO_ETAPA2)
    print(f"\nModelo guardado en {config.MODELO_ETAPA2}")

    evaluacion.graficar_historial(
        historial, config.FIG_DIR / "p2_historial.png",
        ("loss", "accuracy", "auc"), "Paso 2: base congelada")

    # --------------------------------------------------------- evaluacion
    umbral, metricas = evaluacion.evaluar_y_graficar(
        modelo, ds_val, "p2", "Paso 2 (base congelada)")
    evaluacion.guardar_umbral(umbral, metricas)
    print(f"\nUmbral guardado en {config.UMBRAL_JSON}")


if __name__ == "__main__":
    tf.get_logger().setLevel("ERROR")
    main()
