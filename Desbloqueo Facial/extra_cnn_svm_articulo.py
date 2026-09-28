"""
Extra: replica del metodo del articulo de referencia.

Razavian, Azizpour, Sullivan y Carlsson (2014), "CNN Features off-the-shelf:
an Astounding Baseline for Recognition" (arXiv:1403.6382).

Lo que hace el articulo:
  1. Toma una red ya entrenada (OverFeat, entrenada en ImageNet) y NO la
     vuelve a entrenar.
  2. Pasa cada imagen por la red y toma las activaciones de la primera capa
     totalmente conectada (4096 numeros) como "vector de caracteristicas".
  3. Normaliza ese vector a norma L2 = 1.
  4. Entrena un clasificador lineal sencillo (SVM lineal) sobre esos vectores.
  5. Variante "CNNaug-SVM": aumenta el conjunto de entrenamiento con
     recortes, rotaciones y espejos (jittering) y aplica una transformacion
     de potencia componente a componente.
Con eso supera a metodos disenados a mano en tareas muy distintas a la
original (escenas, aves, flores, atributos humanos, busqueda de imagenes).

Aqui hacemos lo mismo para el desbloqueo facial:
  * red pre-entrenada   = base convolucional del paso 1 (CelebA), congelada
  * vector              = salida de GlobalAveragePooling (256 dimensiones)
  * normalizacion L2 + SVM lineal (scikit-learn)
  * CNNaug-SVM          = las fotos de entrenamiento pasan por el mismo
                          aumento aleatorio que en el paso 2

Sirve como linea base para comparar contra el clasificador denso de los
pasos 2 y 3: si la SVM ya logra buenos resultados, confirma que las
caracteristicas aprendidas en CelebA son las que hacen el trabajo pesado.

Uso:
    python extra_cnn_svm_articulo.py
"""

import numpy as np
import tensorflow as tf
import keras
from sklearn.svm import LinearSVC

import config
import datos
import modelos
import evaluacion


def extraer_caracteristicas(base, ds, potencia=1.0):
    """Pasa todo el dataset por la base y regresa (X, y).

    X se normaliza a norma L2 unitaria (como en el articulo). Con
    `potencia != 1` se aplica antes la transformacion de potencia con signo
    sign(x) * |x|^potencia que usa la variante CNNaug-SVM.
    """
    xs, ys = [], []
    for x, y in ds:
        xs.append(base.predict_on_batch(x))
        ys.append(y.numpy().ravel())
    X = np.concatenate(xs)
    if potencia != 1.0:
        X = np.sign(X) * np.abs(X) ** potencia
    X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
    return X, np.concatenate(ys).astype(int)


def main():
    config.crear_carpetas_salida()
    keras.utils.set_random_seed(config.SEMILLA)

    print("=" * 70)
    print("Extra: CNN off-the-shelf + SVM lineal (Razavian et al., 2014)")
    print("=" * 70)

    if not config.MODELO_CELEBA.exists():
        raise FileNotFoundError(f"No existe {config.MODELO_CELEBA}. "
                                "Ejecuta primero paso1_preentrenar_celeba.py")
    base = modelos.extraer_base(keras.models.load_model(config.MODELO_CELEBA))
    base.trainable = False

    # ds_train ya viene aumentado (jittering) y equilibrado
    ds_train, ds_val, resumen = datos.datasets_mi_rostro()
    print(f"Positivos/negativos train: {resumen['positivos_train']}/"
          f"{resumen['negativos_train']}  val: {resumen['positivos_val']}/"
          f"{resumen['negativos_val']}")

    resultados = {}
    for nombre, potencia in (("CNNaug-SVM", 1.0),
                             ("CNNaug-SVM + potencia 2", 2.0)):
        X_tr, y_tr = extraer_caracteristicas(base, ds_train, potencia)
        X_va, y_va = extraer_caracteristicas(base, ds_val, potencia)

        # SVM lineal: busca el hiperplano w.x + b = 0 que separa mis fotos de
        # las de otros con el mayor margen. C controla cuanto se penalizan los
        # errores de entrenamiento (C grande = menos regularizacion).
        svm = LinearSVC(C=1.0, class_weight="balanced", max_iter=20000)
        svm.fit(X_tr, y_tr)

        # decision_function = distancia (con signo) al hiperplano. Umbral
        # natural: 0. Tambien se busca el umbral que cumple FAR_OBJETIVO.
        puntaje = svm.decision_function(X_va)
        m0 = evaluacion.metricas_binarias(y_va, puntaje, 0.0)
        evaluacion.imprimir_metricas(f"{nombre} - umbral 0", m0)
        u = evaluacion.umbral_para_far(y_va, puntaje)
        mu = evaluacion.metricas_binarias(y_va, puntaje, u)
        evaluacion.imprimir_metricas(
            f"{nombre} - umbral para FAR <= {config.FAR_OBJETIVO}", mu)
        resultados[nombre] = m0

        prefijo = "extra_svm" if potencia == 1.0 else "extra_svm_potencia"
        evaluacion.graficar_roc(y_va, puntaje,
                                config.FIG_DIR / f"{prefijo}_roc.png", nombre)
        evaluacion.graficar_histograma_puntajes(
            y_va, puntaje, u, config.FIG_DIR / f"{prefijo}_puntajes.png", nombre)

    print("\nResumen (umbral 0):")
    for nombre, m in resultados.items():
        print(f"  {nombre:28s} acc_bal={m['accuracy_balanceada']:.4f}  "
              f"TAR={m['TAR_recall']:.4f}  FAR={m['FAR']:.4f}")
    print("Compara con las metricas de los pasos 2 y 3.")


if __name__ == "__main__":
    tf.get_logger().setLevel("ERROR")
    main()
