# Redes Neuronales Artificiales

Repositorio de tareas de la materia **Redes Neuronales Artificiales**.
Cada tarea vive en su propia carpeta `Tarea_XX_...`.

## Estructura

```
Redes_Neuronales_Artificiales/
├── README.md
├── requirements.txt
├── .gitignore
└── Tarea_01_MNIST_Densa_Optuna_Regularizacion/
    └── MNIST_Densa_Optuna_MLflow_Regularizacion.ipynb
```

Para cada tarea nueva, crea una carpeta siguiendo el mismo patrón:
`Tarea_02_...`, `Tarea_03_...`, etc.

## Tareas

| # | Carpeta | Descripción |
|---|---------|-------------|
| 01 | `Tarea_01_MNIST_Densa_Optuna_Regularizacion` | Red densa (MNIST) con búsqueda de arquitectura vía Optuna, registro en MLflow y comparación de regularizaciones (L1, L2, L1-L2, Dropout, Dropout+L1-L2). |

## Seguimiento de experimentos (MLflow + DagsHub)

Los experimentos se registran en **MLflow** usando **DagsHub** como servidor remoto,
lo que produce un enlace público donde se pueden ver las gráficas y métricas.

Enlace del servidor:

```
https://dagshub.com/<TU_USUARIO>/Redes_Neuronales_Artificiales/experiments
```

## Instalación

```bash
pip install -r requirements.txt
```
