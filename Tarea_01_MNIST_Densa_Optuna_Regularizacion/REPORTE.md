# Reporte — Red Densa (MNIST) + Optuna + MLflow + Regularización

**Materia:** Redes Neuronales Artificiales
**Autora:** Brenda Mones
**Dataset:** MNIST (70 000 imágenes 28×28 de dígitos 0–9)
**Servidor de experimentos (MLflow / DagsHub):**
<https://dagshub.com/BrendaMonesA/Redes_Neuronales_Artificiales/experiments>

---

## 1. Objetivo y metodología

Se diseñó una **red densa secuencial (no convolucional)** en Keras para clasificar
dígitos MNIST. Cada imagen 28×28 se aplana (`Flatten`) a 784 entradas que alimentan
capas `Dense`, con una capa final `softmax` de 10 salidas.

- **Datos:** normalización a [0, 1]; división 90 % entrenamiento (54 000) / 10 % validación
  (6 000), más el conjunto de prueba (10 000). Semilla fija (`SEED=42`) para reproducibilidad.
- **Pérdida:** `sparse_categorical_crossentropy`. **Métrica:** accuracy.
- **Seguimiento:** cada experimento se registra en **MLflow** con servidor remoto en
  **DagsHub** (parámetros, métricas y comparación de runs).

---

## 2. Inciso (a) — Búsqueda de arquitectura con Optuna (sin regularización)

Se usó **Optuna** (30 trials, 10 épocas por trial) para explorar, **sin regularización**:

| Hiperparámetro | Espacio de búsqueda |
|---|---|
| Nº de capas ocultas | 1 – 4 |
| Neuronas por capa | 32, 64, 128, 256, 512 |
| Activación | relu, tanh, elu |
| Optimizador | adam, rmsprop, sgd |
| Learning rate | 1e-4 a 1e-2 (escala log) |
| Batch size | 32, 64, 128 |

Cada trial quedó registrado como un run en MLflow (`optuna_trial_0` … `optuna_trial_29`).

### Mejor arquitectura encontrada (trial 22)

| Parámetro | Valor |
|---|---|
| Capas ocultas | 2 |
| Neuronas | 256 → 256 |
| Activaciones | relu → tanh |
| Optimizador | RMSprop |
| Learning rate | 0.00116 |
| Batch size | 32 |
| **Validation accuracy** | **0.9788** |

**Observaciones.** Las mejores configuraciones tendieron a **redes de 2–4 capas con
128–256 neuronas y RMSprop/Adam**; SGD y los learning rates altos (>3e-3) dieron los
peores resultados (p. ej. trial 2 con SGD cayó a 0.817 de val_accuracy). La arquitectura
ganadora es relativamente **compacta** (2 capas) y ya generaliza muy bien.

---

## 3. Inciso (b) — Regularización sobre la mejor arquitectura

Se reentrenó la mejor arquitectura (20 épocas) con los mismos datos, variando la
regularización. Se incluye el modelo **Base** (sin regularización) como referencia.
Regularizadores L1/L2/L1-L2 con λ = 1e-4; Dropout con tasa 0.30.

### Tabla comparativa (resultados reales)

| Modelo | Train Acc | Val Acc | **Test Acc** | Test Loss | Min Val Loss | Brecha Train−Val |
|---|---|---|---|---|---|---|
| **Dropout** | 0.9900 | 0.9810 | **0.9835** | 0.0775 | 0.0875 | 0.0090 |
| Base | 0.9998 | 0.9813 | 0.9825 | 0.1242 | 0.1048 | 0.0185 |
| L1 | 0.9781 | 0.9680 | 0.9660 | 0.2012 | 0.2046 | 0.0101 |
| L2 | 0.9877 | 0.9713 | 0.9627 | 0.1721 | 0.1574 | 0.0163 |
| L1-L2 | 0.9765 | 0.9613 | 0.9626 | 0.2116 | 0.2333 | 0.0152 |
| Dropout + L1-L2 | 0.9473 | 0.9633 | 0.9619 | 0.2669 | 0.2722 | −0.0160 |

> **Nota sobre `Test Loss`:** en L1/L2/L1-L2 la pérdida **incluye el término de penalización**
> de los pesos, por lo que no es directamente comparable con la de Base/Dropout. La comparación
> más justa entre modelos es la de **accuracy** y la **brecha train−val**.

Figuras generadas (carpeta `figuras/`):
- `val_accuracy.png` — Validation Accuracy por modelo.
- `val_loss.png` — Validation Loss por modelo.
- `base_train_vs_val.png` — Modelo base: Train vs Validation (accuracy y loss).
- `test_accuracy.png` — Test Accuracy por modelo.

### Comentario de cada caso

- **Base (sin regularización).** Alcanza **train ≈ 0.9998** (memoriza casi todo el
  entrenamiento) con **val 0.9813** y **test 0.9825**. La brecha train−val (0.0185) es la mayor
  de los modelos "buenos": hay un **ligero sobreajuste**, pero el desempeño en test sigue siendo alto.

- **L1 (λ=1e-4).** Reduce la brecha a 0.0101, pero **baja el desempeño**: val 0.968 y **test 0.966**.
  La penalización sobre el valor absoluto de los pesos fue **demasiado fuerte** para esta red que ya
  generalizaba bien → tiende a *underfitting*. No mejoró la eficiencia.

- **L2 (λ=1e-4).** Comportamiento intermedio: brecha 0.0163 y **test 0.9627**. Suaviza los pesos
  pero también **sacrifica accuracy** frente a Base. Tampoco mejoró el resultado final.

- **L1-L2 (λ=1e-4 c/u).** Combina ambas penalizaciones; brecha 0.0152 y **test 0.9626**. Resultado
  similar a L1/L2: reduce el sobreajuste pero **empeora el desempeño**.

- **Dropout (0.30).** **El mejor modelo:** **test 0.9835** (supera a Base), brecha pequeña (0.0090)
  y la **menor Test Loss** (0.0775). Al desactivar neuronas aleatoriamente durante el entrenamiento,
  **redujo el sobreajuste y mejoró la generalización** sin sacrificar accuracy.

- **Dropout + L1-L2.** Regularización **excesiva**: brecha **negativa** (−0.016, val > train) que
  indica **underfitting**, y el **peor test (0.9619)** con la mayor pérdida. Apilar Dropout con L1-L2
  penalizó demasiado al modelo.

---

## 4. Pregunta principal

> **¿La regularización ayudó a mejorar la eficiencia antes de haber sobreajuste?**

**Depende de la técnica; en este caso solo el Dropout ayudó.**

- El modelo **Base ya generalizaba muy bien** (test 0.9825) con un sobreajuste apenas leve
  (train casi 1.0 frente a val 0.981). Cuando el punto de partida no sobreajusta fuerte, agregar
  penalizaciones agresivas tiende a **perjudicar**.
- **Dropout (0.30)** fue la única técnica que **mejoró la eficiencia**: subió el test (0.9835 vs 0.9825),
  redujo la brecha train−val (0.0090 vs 0.0185) y bajó la pérdida. Es decir, **redujo el sobreajuste
  y a la vez mejoró la generalización**.
- **L1, L2 y L1-L2** con λ=1e-4 **redujeron la brecha** pero **a costa del desempeño** (test ~0.962–0.966):
  provocaron un ligero *underfitting*. Reducir la brecha **no equivale** a mejorar el modelo.
- **Dropout + L1-L2** fue **contraproducente**: demasiada regularización → underfitting y el peor test.

**Conclusión.** Para esta red densa sobre MNIST, la regularización **más efectiva fue Dropout**.
Las penalizaciones de pesos (L1/L2/L1-L2) con la intensidad usada resultaron excesivas y bajaron
el desempeño. Una posible mejora sería **ajustar λ** (valores más pequeños, p. ej. 1e-5) o
**entrenar más épocas** con L1/L2 para permitir que converjan; sin embargo, con la configuración
evaluada, **Dropout es la mejor opción**.

---

## 5. Reproducibilidad

```bash
pip install -r requirements.txt
cd Tarea_01_MNIST_Densa_Optuna_Regularizacion
python mnist_densa_optuna_regularizacion.py
```

El token de DagsHub se lee desde un archivo `.env` local (no versionado). Todos los runs
—30 trials de Optuna + 6 modelos de regularización— quedan en el servidor:
<https://dagshub.com/BrendaMonesA/Redes_Neuronales_Artificiales/experiments>
