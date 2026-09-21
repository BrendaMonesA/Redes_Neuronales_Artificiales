# Redes Neuronales Artificiales

Repositorio de tareas de la materia **Redes Neuronales Artificiales**.
Cada tarea vive en su propia carpeta `Tarea_XX_...`.

## Tareas

| # | Carpeta | Descripción |
|---|---------|-------------|
| 01 | `Tarea_01_MNIST_Densa_Optuna_Regularizacion` | Red densa (MNIST): búsqueda de arquitectura con Optuna (sin regularización), registro en MLflow, y comparación de regularizaciones L1, L2, L1-L2, Dropout y Dropout+L1-L2. |

## Seguimiento de experimentos (MLflow + DagsHub)

Los experimentos se registran en **MLflow**. Para tener un enlace público donde ver
las gráficas se usa **DagsHub** como servidor MLflow remoto:

```
https://dagshub.com/<TU_USUARIO>/Redes_Neuronales_Artificiales/experiments
```

## Instalación

```bash
pip install -r requirements.txt
```

## Uso

Abre el notebook de la tarea (Colab o local), configura tu token de DagsHub en la
celda 5 y ejecuta todas las celdas.
