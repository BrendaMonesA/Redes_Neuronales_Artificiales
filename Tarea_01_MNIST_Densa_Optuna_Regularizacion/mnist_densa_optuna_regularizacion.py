"""
Tarea 01 - Redes Neuronales Artificiales
=========================================
Red densa secuencial (no convolucional) para clasificacion de digitos MNIST.

(a) Busqueda de arquitectura con Optuna (SIN regularizacion), registrada en MLflow.
(b) Entrenamiento de la mejor arquitectura con regularizaciones:
    Base, L1, L2, L1-L2, Dropout, Dropout+L1-L2 (registrado en MLflow).

Ejecutar:
    pip install -r requirements.txt
    python mnist_densa_optuna_regularizacion.py

El token de DagsHub se lee desde un archivo .env (NO se sube al repositorio).
Si no hay token, MLflow registra en local (./mlruns).
"""

import os
import random
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")  # backend sin ventana: guarda figuras a archivo
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers

import optuna
import mlflow

from dotenv import load_dotenv

# ------------------------------------------------------------------
# Configuracion general
# ------------------------------------------------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

N_TRIALS = 30          # trials de Optuna
EPOCHS_OPTUNA = 10     # epocas por trial
EPOCHS_FINAL = 20      # epocas de los modelos finales

FIG_DIR = "figuras"
os.makedirs(FIG_DIR, exist_ok=True)

# DagsHub / MLflow
DAGSHUB_USER = "BrendaMonesA"
DAGSHUB_REPO = "Redes_Neuronales_Artificiales"


# ------------------------------------------------------------------
# 1. MLflow (DagsHub remoto si hay token; si no, local)
# ------------------------------------------------------------------
def configurar_mlflow():
    load_dotenv()  # carga variables del archivo .env
    token = os.environ.get("DAGSHUB_TOKEN", "")

    if token:
        os.environ["MLFLOW_TRACKING_USERNAME"] = DAGSHUB_USER
        os.environ["MLFLOW_TRACKING_PASSWORD"] = token
        mlflow.set_tracking_uri(f"https://dagshub.com/{DAGSHUB_USER}/{DAGSHUB_REPO}.mlflow")
        print("MLflow -> DagsHub (remoto).")
        print("ENLACE DEL SERVIDOR (para el reporte):")
        print(f"  https://dagshub.com/{DAGSHUB_USER}/{DAGSHUB_REPO}/experiments")
    else:
        mlflow.set_tracking_uri("file:./mlruns")
        print("MLflow -> local ./mlruns  (define DAGSHUB_TOKEN en .env para el enlace publico)")

    mlflow.set_experiment("MNIST_Dense_Optuna")
    print("Tracking URI:", mlflow.get_tracking_uri())


# ------------------------------------------------------------------
# 2. Datos
# ------------------------------------------------------------------
def cargar_datos():
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    rng = np.random.default_rng(SEED)
    idx = rng.permutation(len(x_train))
    n_val = int(0.10 * len(x_train))
    val_idx, train_idx = idx[:n_val], idx[n_val:]

    x_val, y_val = x_train[val_idx], y_train[val_idx]
    x_tr, y_tr = x_train[train_idx], y_train[train_idx]

    print("Train:", x_tr.shape, "| Val:", x_val.shape, "| Test:", x_test.shape)
    return (x_tr, y_tr), (x_val, y_val), (x_test, y_test)


# ------------------------------------------------------------------
# 3. Constructor de la red densa
# ------------------------------------------------------------------
def build_dense_model(n_layers, units, activations, optimizer_name,
                      learning_rate, regularization=None, dropout_rate=0.0):
    """regularization: None | 'l1' | 'l2' | 'l1_l2'.  dropout_rate=0 desactiva Dropout."""
    model = keras.Sequential(name="MNIST_Dense")
    model.add(layers.Input(shape=(28, 28)))
    model.add(layers.Flatten())

    for i in range(n_layers):
        reg = None
        if regularization == "l1":
            reg = regularizers.l1(1e-4)
        elif regularization == "l2":
            reg = regularizers.l2(1e-4)
        elif regularization == "l1_l2":
            reg = regularizers.l1_l2(l1=1e-4, l2=1e-4)

        model.add(layers.Dense(units[i], activation=activations[i], kernel_regularizer=reg))
        if dropout_rate > 0:
            model.add(layers.Dropout(dropout_rate))

    model.add(layers.Dense(10, activation="softmax"))

    opt = {
        "adam": keras.optimizers.Adam(learning_rate=learning_rate),
        "rmsprop": keras.optimizers.RMSprop(learning_rate=learning_rate),
        "sgd": keras.optimizers.SGD(learning_rate=learning_rate),
    }[optimizer_name]

    model.compile(optimizer=opt, loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


# ------------------------------------------------------------------
# 4. Optuna - inciso (a): SIN regularizacion
# ------------------------------------------------------------------
def make_objective(data):
    (x_tr, y_tr), (x_val, y_val), _ = data

    def objective(trial):
        n_layers = trial.suggest_int("n_layers", 1, 4)
        units = [trial.suggest_categorical(f"units_{i}", [32, 64, 128, 256, 512])
                 for i in range(n_layers)]
        activations = [trial.suggest_categorical(f"activation_{i}", ["relu", "tanh", "elu"])
                       for i in range(n_layers)]
        optimizer_name = trial.suggest_categorical("optimizer", ["adam", "rmsprop", "sgd"])
        learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)
        batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])

        model = build_dense_model(n_layers, units, activations, optimizer_name, learning_rate)

        with mlflow.start_run(run_name=f"optuna_trial_{trial.number}"):
            mlflow.log_params(trial.params)
            history = model.fit(x_tr, y_tr, validation_data=(x_val, y_val),
                                epochs=EPOCHS_OPTUNA, batch_size=batch_size, verbose=0)
            best_val_acc = float(max(history.history["val_accuracy"]))
            mlflow.log_metric("best_val_accuracy", best_val_acc)
            mlflow.log_metric("final_val_loss", float(history.history["val_loss"][-1]))

        keras.backend.clear_session()
        return best_val_acc

    return objective


def buscar_arquitectura(data):
    study = optuna.create_study(direction="maximize", study_name="MNIST_Dense_Search")
    study.optimize(make_objective(data), n_trials=N_TRIALS, show_progress_bar=True)

    print("\nMejor validation accuracy:", study.best_value)
    print("Mejores hiperparametros:")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")

    study.trials_dataframe().to_csv("optuna_trials.csv", index=False)
    return study


# ------------------------------------------------------------------
# 5. Entrenamiento final - inciso (b)
# ------------------------------------------------------------------
def entrenar_regularizaciones(study, data):
    (x_tr, y_tr), (x_val, y_val), (x_test, y_test) = data
    bp = study.best_params
    n_layers = bp["n_layers"]
    units = [bp[f"units_{i}"] for i in range(n_layers)]
    activations = [bp[f"activation_{i}"] for i in range(n_layers)]
    optimizer_name = bp["optimizer"]
    lr = bp["learning_rate"]
    batch_size = bp["batch_size"]

    configs = {
        "Base": (None, 0.0),
        "L1": ("l1", 0.0),
        "L2": ("l2", 0.0),
        "L1-L2": ("l1_l2", 0.0),
        "Dropout": (None, 0.30),
        "Dropout + L1-L2": ("l1_l2", 0.30),
    }

    results, models = {}, {}
    for name, (reg, dr) in configs.items():
        print(f"\n===== {name} =====")
        keras.backend.clear_session()
        model = build_dense_model(n_layers, units, activations, optimizer_name, lr,
                                  regularization=reg, dropout_rate=dr)

        with mlflow.start_run(run_name=name):
            mlflow.log_param("model_type", name)
            mlflow.log_param("regularization", str(reg))
            mlflow.log_param("dropout_rate", dr)
            history = model.fit(x_tr, y_tr, validation_data=(x_val, y_val),
                                epochs=EPOCHS_FINAL, batch_size=batch_size, verbose=1)
            test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
            mlflow.log_metric("test_loss", float(test_loss))
            mlflow.log_metric("test_accuracy", float(test_acc))

        models[name] = model
        results[name] = {"history": history, "test_loss": test_loss, "test_accuracy": test_acc}

    return results, models


# ------------------------------------------------------------------
# 6. Reportes: tablas, figuras y modelos
# ------------------------------------------------------------------
def generar_reportes(results, models):
    rows = []
    for name, r in results.items():
        h = r["history"].history
        train_acc = max(h["accuracy"])
        val_acc = max(h["val_accuracy"])
        rows.append({
            "Modelo": name,
            "Mejor Train Acc": train_acc,
            "Mejor Val Acc": val_acc,
            "Test Acc": r["test_accuracy"],
            "Test Loss": r["test_loss"],
            "Min Val Loss": min(h["val_loss"]),
            "Brecha Train-Val": train_acc - val_acc,
        })
    results_df = pd.DataFrame(rows).sort_values("Test Acc", ascending=False)
    results_df.to_csv("resultados_regularizacion.csv", index=False)

    print("\n===== TABLA COMPARATIVA =====")
    print(results_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    # Validation Accuracy
    plt.figure(figsize=(12, 7))
    for name, r in results.items():
        plt.plot(r["history"].history["val_accuracy"], label=name)
    plt.xlabel("Epoca"); plt.ylabel("Validation Accuracy")
    plt.title("Validation Accuracy por modelo"); plt.legend(); plt.grid(True, alpha=0.3)
    plt.savefig(f"{FIG_DIR}/val_accuracy.png", dpi=120, bbox_inches="tight"); plt.close()

    # Validation Loss
    plt.figure(figsize=(12, 7))
    for name, r in results.items():
        plt.plot(r["history"].history["val_loss"], label=name)
    plt.xlabel("Epoca"); plt.ylabel("Validation Loss")
    plt.title("Validation Loss por modelo"); plt.legend(); plt.grid(True, alpha=0.3)
    plt.savefig(f"{FIG_DIR}/val_loss.png", dpi=120, bbox_inches="tight"); plt.close()

    # Base: train vs val
    h = results["Base"]["history"].history
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    ax[0].plot(h["accuracy"], label="Train"); ax[0].plot(h["val_accuracy"], label="Val")
    ax[0].set_title("Base: Accuracy"); ax[0].set_xlabel("Epoca"); ax[0].legend(); ax[0].grid(True, alpha=0.3)
    ax[1].plot(h["loss"], label="Train"); ax[1].plot(h["val_loss"], label="Val")
    ax[1].set_title("Base: Loss"); ax[1].set_xlabel("Epoca"); ax[1].legend(); ax[1].grid(True, alpha=0.3)
    plt.savefig(f"{FIG_DIR}/base_train_vs_val.png", dpi=120, bbox_inches="tight"); plt.close()

    # Test Accuracy (barras)
    plot_df = results_df.sort_values("Test Acc")
    plt.figure(figsize=(10, 6))
    plt.barh(plot_df["Modelo"], plot_df["Test Acc"])
    plt.xlabel("Test Accuracy"); plt.title("Test Accuracy por modelo")
    plt.xlim(max(0, plot_df["Test Acc"].min() - 0.02), min(1, plot_df["Test Acc"].max() + 0.01))
    plt.grid(axis="x", alpha=0.3)
    plt.savefig(f"{FIG_DIR}/test_accuracy.png", dpi=120, bbox_inches="tight"); plt.close()

    # Modelos
    os.makedirs("modelos_mnist", exist_ok=True)
    for name, model in models.items():
        safe = name.lower().replace(" ", "_").replace("+", "plus")
        model.save(f"modelos_mnist/{safe}.keras")

    print(f"\nFiguras guardadas en '{FIG_DIR}/'")
    print("CSV: resultados_regularizacion.csv, optuna_trials.csv")
    print("Modelos: modelos_mnist/")
    return results_df


# ------------------------------------------------------------------
# main
# ------------------------------------------------------------------
def main():
    print("TensorFlow:", tf.__version__, "| Optuna:", optuna.__version__, "| MLflow:", mlflow.__version__)
    configurar_mlflow()
    data = cargar_datos()

    print("\n--- (a) Busqueda de arquitectura con Optuna (sin regularizacion) ---")
    study = buscar_arquitectura(data)

    print("\n--- (b) Entrenamiento con regularizaciones ---")
    results, models = entrenar_regularizaciones(study, data)

    generar_reportes(results, models)
    print("\nListo. Revisa las figuras y el enlace de MLflow para el reporte.")


if __name__ == "__main__":
    main()
