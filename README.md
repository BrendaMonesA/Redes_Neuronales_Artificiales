# Redes Neuronales Artificiales

Repositorio de tareas de la materia **Redes Neuronales Artificiales**.
Cada tarea vive en su propia carpeta `Tarea_XX_...`.

## Tareas

| # | Carpeta | Descripción |
|---|---------|-------------|
| 01 | `Tarea_01_MNIST_Densa_Optuna_Regularizacion` | Red densa (MNIST): búsqueda de arquitectura con Optuna (sin regularización) + comparación de regularizaciones L1, L2, L1-L2, Dropout y Dropout+L1-L2. Registro en MLflow (DagsHub). |

## Requisitos

```bash
pip install -r requirements.txt
```

## Configurar el token de DagsHub (MLflow)

El token **no se guarda en el repositorio**. Se lee desde un archivo `.env` local
(incluido en `.gitignore`).

1. Copia `Tarea_01_.../.env.example` como `.env` en la misma carpeta.
2. Pega tu token real:
   ```
   DAGSHUB_TOKEN=tu_token_de_dagshub
   ```

Si no configuras el token, el script registra en MLflow local (`./mlruns`).

## Ejecutar la Tarea 01

```bash
cd Tarea_01_MNIST_Densa_Optuna_Regularizacion
python mnist_densa_optuna_regularizacion.py
```

Genera:
- `resultados_regularizacion.csv`, `optuna_trials.csv`
- Figuras en `figuras/`
- Modelos en `modelos_mnist/`
- Runs en MLflow: `https://dagshub.com/BrendaMonesA/Redes_Neuronales_Artificiales/experiments`
