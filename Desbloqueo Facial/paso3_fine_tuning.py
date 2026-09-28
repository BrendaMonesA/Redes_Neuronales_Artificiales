"""
Paso 3 (estrategia 2.2.3c): fine tuning / refinamiento.

Ya con el clasificador del paso 2 dando valores razonables, ahora si se
descongelan las ULTIMAS capas convolucionales y se entrenan junto con las
capas densas, pero con un learning rate muy pequeno (1e-5, 100 veces menor
que en el paso 2).

Por que funciona:
* Los ultimos bloques convolucionales codifican rasgos de alto nivel que en
  el paso 1 se ajustaron para los atributos de CelebA. Moverlos un poco
  permite especializarlos en lo que distingue MI cara.
* El learning rate pequeno evita "olvidar" lo aprendido (olvido
  catastrofico): solo se hacen ajustes finos alrededor de los pesos
  pre-entrenados.
* Los primeros bloques (bordes, texturas) y las BatchNormalization se
  mantienen congelados.

Al terminar se elige el umbral de decision que cumple FAR <= FAR_OBJETIVO
en validacion y se guarda para desbloquear.py.

Salida:
    modelos/rostro_final.keras
    modelos/umbral.json
    figuras/p3_historial.png, p3_roc.png, p3_confusion.png, p3_puntajes.png

Uso:
    python paso3_fine_tuning.py
    python paso3_fine_tuning.py --bloques 2
"""

import argparse

import tensorflow as tf
import keras

import config
import datos
import modelos
import evaluacion


def argumentos():
    p = argparse.ArgumentParser(description="Paso 3: fine tuning")
    p.add_argument("--epocas", type=int, default=config.EPOCAS_ETAPA3)
    p.add_argument("--bloques", type=int, default=config.BLOQUES_A_DESCONGELAR,
                   help="bloques convolucionales finales a descongelar")
    p.add_argument("--lr", type=float, default=config.LR_ETAPA3)
    return p.parse_args()


def main():
    args = argumentos()
    config.crear_carpetas_salida()
    keras.utils.set_random_seed(config.SEMILLA)

    print("=" * 70)
    print("Paso 3: fine tuning de las ultimas capas convolucionales + densas")
    print("=" * 70)

    ds_train, ds_val, resumen = datos.datasets_mi_rostro()
    for clave, valor in resumen.items():
        print(f"  {clave:18s}: {valor}")

    if not config.MODELO_ETAPA2.exists():
        raise FileNotFoundError(f"No existe {config.MODELO_ETAPA2}. "
                                "Ejecuta primero paso2_entrenar_clasificador.py")
    modelo = keras.models.load_model(config.MODELO_ETAPA2)

    # Punto de partida: como quedo el modelo del paso 2
    print("\nAntes del fine tuning:")
    antes = modelo.evaluate(ds_val, return_dict=True, verbose=0)
    print("  " + "  ".join(f"{k}={v:.4f}" for k, v in antes.items()))

    # Descongelar los ultimos bloques (y recompilar: obligatorio despues de
    # cambiar `trainable`, si no Keras seguiria usando la lista vieja).
    base = modelos.extraer_base(modelo)
    bloques = modelos.descongelar_ultimos_bloques(base, args.bloques)
    entrenables, congelados = modelos.contar_parametros(modelo)
    print(f"\nBloques descongelados: {', '.join(bloques)}")
    print(f"Parametros entrenables: {entrenables:,}  congelados: {congelados:,}")
    for capa in base.layers:
        if capa.weights:
            print(f"  {capa.name:18s} {'ENTRENA' if capa.trainable else 'congelada'}")

    modelos.compilar_binario(modelo, args.lr)
    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=4,
                                      restore_best_weights=True, verbose=1),
    ]
    historial = modelo.fit(ds_train, validation_data=ds_val,
                           epochs=args.epocas, callbacks=callbacks)
    modelo.save(config.MODELO_FINAL)
    print(f"\nModelo guardado en {config.MODELO_FINAL}")

    evaluacion.graficar_historial(
        historial, config.FIG_DIR / "p3_historial.png",
        ("loss", "accuracy", "auc"), "Paso 3: fine tuning")

    print("\nDespues del fine tuning:")
    despues = modelo.evaluate(ds_val, return_dict=True, verbose=0)
    print("  " + "  ".join(f"{k}={v:.4f}" for k, v in despues.items()))

    umbral, metricas = evaluacion.evaluar_y_graficar(
        modelo, ds_val, "p3", "Paso 3 (fine tuning)")
    evaluacion.guardar_umbral(umbral, metricas)
    print(f"\nUmbral final {umbral:.4f} guardado en {config.UMBRAL_JSON}")
    print("Nota: el umbral se elige sobre validacion; con pocas fotos conviene "
          "probarlo con fotos nuevas usando desbloquear.py.")


if __name__ == "__main__":
    tf.get_logger().setLevel("ERROR")
    main()
