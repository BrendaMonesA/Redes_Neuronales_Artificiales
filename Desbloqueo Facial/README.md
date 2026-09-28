# Tarea: Desbloqueo facial con transfer learning

Entrenar una red neuronal que reconozca **mi rostro** (como el desbloqueo facial
de un celular) teniendo muy pocas fotos mías. Para lograrlo se usa
**transfer learning**: primero se entrena una CNN con muchas caras de otras
personas (CelebA) y después se reutilizan sus capas convolucionales para la
tarea "¿soy yo o no?".

Referencia: A. S. Razavian, H. Azizpour, J. Sullivan, S. Carlsson (2014),
*CNN Features off-the-shelf: an Astounding Baseline for Recognition*,
arXiv:1403.6382.

---

## 1. Idea general

### ¿Por qué transfer learning?

Una CNN con millones de parámetros necesita decenas de miles de imágenes para
entrenarse desde cero. De mí hay, con suerte, unas decenas de fotos. Pero:

* Las **primeras capas** de una CNN aprenden cosas genéricas (bordes, colores,
  texturas) que sirven para cualquier imagen.
* Las **capas intermedias y finales** aprenden partes (ojos, boca, nariz,
  cabello) si se entrenaron con rostros.

El artículo de referencia mostró que las activaciones de una red ya entrenada
(OverFeat en ImageNet), usadas tal cual como vector de características y con un
SVM lineal encima, superan a métodos diseñados a mano en tareas muy distintas
a la original: escenas, aves, flores, atributos humanos y búsqueda de imágenes.
Si eso funciona para tareas lejanas, para "reconocer una cara" con una red
entrenada en caras debería funcionar todavía mejor.

### Las 3 etapas del reconocimiento facial

1. **Detección**: encontrar dónde hay caras en la imagen.
2. **Alineación**: recortar y normalizar la cara a un tamaño fijo.
3. **Extracción de características**: convertir la cara en un vector que la describe.

La tarea solo pide la **etapa 3**: la red recibe fotos que ya son caras.
(El script opcional `capturar_fotos.py` resuelve la 1 y la 2 de forma sencilla
con OpenCV para armar el conjunto de fotos.)

### Estrategia (tal como la sugiere el enunciado)

```
Paso 1  CelebA (≈200k caras, 40 atributos)
        imagen ──► [BASE CONVOLUCIONAL] ──► [Densa 256] ──► 40 sigmoides
                                                   │
Paso 2  se quita el clasificador de atributos      ▼  (se descarta)
        imagen ──► [BASE CONVOLUCIONAL] ──► [Densa 128] ──► 1 sigmoide = P(soy yo)
                    CONGELADA                 entrena        entrena

Paso 3  fine tuning con learning rate muy pequeño
        imagen ──► [bloques 1-3 | bloque 4] ──► [Densa 128] ──► 1 sigmoide
                    congelados   entrena         entrena
```

---

## 2. Archivos

| Archivo | Qué hace |
|---|---|
| `config.py` | Rutas e hiperparámetros compartidos (tamaño de imagen, épocas, learning rates, FAR objetivo...). |
| `datos.py` | Lectura de CelebA con `tfds.load("celeb_a")` (o desde archivos locales como respaldo), de mis fotos y de los negativos; preprocesamiento (recorte cuadrado, 128×128, escala [0,1]); aumento de datos; armado de conjuntos equilibrados. |
| `modelos.py` | Arquitectura de la CNN, construcción del clasificador nuevo, funciones para **congelar** / **descongelar** capas. |
| `evaluacion.py` | Métricas biométricas (TAR, FAR, FRR), elección del umbral, gráficas (curvas, ROC, matriz de confusión, histogramas). |
| `paso1_preentrenar_celeba.py` | **Entrenamiento 1**: CNN que predice los 40 atributos de CelebA. |
| `paso2_entrenar_clasificador.py` | **Entrenamiento 2**: quita el clasificador, agrega uno de 1 neurona, congela la base y entrena. |
| `paso3_fine_tuning.py` | **Entrenamiento 3**: descongela el último bloque convolucional y refina todo con LR pequeño. |
| `extra_cnn_svm_articulo.py` | Réplica del método del artículo: características "off-the-shelf" + normalización L2 + SVM lineal. |
| `desbloquear.py` | Usa el modelo final para decir **DESBLOQUEADO / BLOQUEADO** sobre fotos nuevas. |
| `capturar_fotos.py` | (Opcional, requiere `opencv-python`) Toma fotos con la webcam o recorta caras de una carpeta. |

---

## 3. Preparar los datos

### CelebA: TensorFlow Datasets (automático)

CelebA se carga con [TensorFlow Datasets](https://www.tensorflow.org/datasets/catalog/celeb_a):

```python
import tensorflow_datasets as tfds
ds = tfds.load("celeb_a", split="train")   # también "validation" y "test"
```

No hay que descargar nada a mano. La primera vez que se corre el paso 1,
`tfds` descarga unos **1.4 GB** y prepara el conjunto (**1.6 GB**) en
`datos/tensorflow_datasets/`. Las siguientes veces lo lee de ahí. Cada ejemplo
es un diccionario:

| Clave | Contenido | ¿Se usa? |
|---|---|---|
| `image` | cara alineada, 218×178×3, `uint8` | sí |
| `attributes` | diccionario con los 40 atributos booleanos | sí (salida del paso 1) |
| `landmarks` | 5 puntos de referencia (ojos, nariz, comisuras) | no |
| `identity` | número de identidad (10,177 personas) | no |

Se usan las tres particiones oficiales: `train` (162,770 imágenes) para
entrenar, `validation` (19,867) para el *early stopping* y `test` (19,962) para
evaluar el paso 1 y para sacar los rostros "de los demás" en los pasos 2 y 3.
Para limitar el tamaño se usa la sintaxis de TFDS `split="train[:40000]"`.

> **Si la descarga falla:** los archivos de CelebA están en Google Drive y a
> veces aparece un error de cuota de descargas. En ese caso se puede esperar y
> reintentar, o descargar CelebA a mano (de la
> [página oficial](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) o de
> [Kaggle](https://www.kaggle.com/datasets/jessicali9530/celeba-dataset)),
> dejarla en `datos/celeba/` y poner `FUENTE_CELEBA = "local"` en `config.py`.
> Ese modo acepta el formato oficial `.txt` y el `.csv` de Kaggle.

### Mis fotos y los negativos

```
Desbloqueo Facial/
└── datos/
    ├── tensorflow_datasets/         ← lo crea tfds automáticamente
    ├── mi_rostro/                   ← MIS fotos (recortadas a la cara)
    └── otros_rostros/               ← (opcional) fotos de otras personas
```

* **Mis fotos**: 30 o más, recortadas a la cara, con distinta luz, expresión,
  ángulo, con/sin lentes. Se pueden tomar con:
  ```bash
  pip install opencv-python
  python capturar_fotos.py
  ```
  o recortar fotos que ya estén en el celular:
  ```bash
  python capturar_fotos.py --recortar-carpeta ruta/a/mis_fotos
  ```
* **Otros rostros** (opcional pero recomendable): fotos de amigos o familiares
  tomadas con la misma cámara. Son los negativos más difíciles y los más
  útiles. Si no hay, los negativos salen solo de la partición `test` de CelebA.

La carpeta `datos/` está en `.gitignore`: ni CelebA ni las fotos personales se
suben al repositorio.

---

## 4. Ejecutar

Desde esta carpeta, con el entorno del repositorio (`tensorflow`, `keras`,
`tensorflow-datasets`, `scikit-learn`, `matplotlib`):

```bash
python paso1_preentrenar_celeba.py
```
```bash
python paso2_entrenar_clasificador.py
```
```bash
python paso3_fine_tuning.py
```
```bash
python extra_cnn_svm_articulo.py
```
```bash
python desbloquear.py ruta/a/foto_nueva.jpg
```

Los modelos quedan en `modelos/` y las gráficas en `figuras/`.
El paso 1 es el más costoso: la primera vez descarga CelebA y por defecto
entrena con 40,000 imágenes (`CELEBA_MAX_TRAIN` en `config.py`; `--max-train 0`
usa las 162,770). En CPU se puede
reducir `IMG_SIZE` a 96 o 64 para que sea más rápido.

---

## 5. Explicación de cada decisión

### Preprocesamiento (`datos.cargar_imagen`)
Las imágenes alineadas de CelebA miden 178×218 con la cara centrada. Se recorta
el **cuadrado central** (178×178) para no deformar la cara al redimensionar y
se lleva a **128×128**, con valores en **[0, 1]**. A mis fotos se les aplica
exactamente el mismo preprocesamiento: la red debe ver ambas fuentes igual.

### Paso 1: ¿por qué predecir atributos?
CelebA no trae "identidad → yo", pero sí 40 atributos por cara (`Smiling`,
`Eyeglasses`, `Male`, `Young`, `Wavy_Hair`, `Big_Nose`...). Para predecirlos
la red se ve obligada a aprender los rasgos del rostro, y esos rasgos son los
que distinguen a una persona de otra.

* **Salida**: 40 neuronas con **sigmoide**, no softmax, porque es
  *multi-etiqueta*: una cara puede tener varios atributos a la vez.
* **Pérdida**: entropía cruzada binaria (una por atributo, promediada).
* **Métricas**: accuracy binaria y AUC multi-etiqueta. La gráfica
  `p1_accuracy_atributos.png` compara cada atributo contra su *línea base*
  (responder siempre la clase mayoritaria): la red debe superarla.
* **Arquitectura** (`modelos.py`): 4 bloques tipo VGG
  `(Conv3×3 → BatchNorm → ReLU) ×2 → MaxPool`, con 32/64/128/256 filtros, y
  al final **Global Average Pooling**, que da un vector de 256 números: la
  "huella" de la cara. Es el equivalente al vector de 4096 dimensiones de la
  capa 22 de OverFeat que usa el artículo.
* La base convolucional es un **sub-modelo con nombre** `base_convolucional`,
  así "quitar el clasificador" es simplemente
  `modelo.get_layer("base_convolucional")`.

### Paso 2: congelar y entrenar el clasificador nuevo
1. Se carga `cnn_celeba.keras` y se extrae la base (se tiran las capas densas).
2. Se agrega `Densa(128, relu) → Dropout(0.4) → Densa(1, sigmoide)`.
3. `base.trainable = False` **antes de compilar**: Keras decide qué pesos
   optimizar en `compile()`.

¿Por qué congelar? Las capas nuevas empiezan con pesos aleatorios y en las
primeras épocas producen gradientes grandes y ruidosos. Si la base estuviera
descongelada, esos gradientes "arruinarían" lo aprendido con CelebA. Primero
se deja que el clasificador llegue a valores razonables.

**Datos equilibrados**: mis fotos de entrenamiento se repiten
`REPETICIONES_POS = 20` veces, y en cada repetición el aumento de datos genera
una versión distinta. Se usan tantos negativos como positivos (50/50), así la
red no puede ganar diciendo siempre "no soy yo".

**Separación train/val por archivo**: primero se separan mis fotos (80/20) y
*después* se aumentan. Si una foto estuviera en train y una versión girada de
ella en validación, la validación sería engañosamente buena.

### Aumento de datos (en lugar de `ImageDataGenerator`)
El enunciado sugiere `ImageDataGenerator`, pero **esa clase ya no existe en
Keras 3** (la versión instalada es 3.15). Su reemplazo son las capas de
preprocesamiento, que hacen lo mismo:

| `ImageDataGenerator` | Keras 3 |
|---|---|
| `horizontal_flip=True` | `RandomFlip("horizontal")` |
| `rotation_range=18` | `RandomRotation(0.05)` (0.05 × 360° = 18°) |
| `zoom_range=0.15` | `RandomZoom(0.15)` |
| `width/height_shift_range=0.1` | `RandomTranslation(0.1, 0.1)` |
| `brightness_range` | `RandomBrightness(0.2)` |
| — | `RandomContrast(0.3)` |

Se aplican dentro del `tf.data` solo al entrenamiento. La figura
`p2_ejemplos_aumento.png` muestra cómo se ven. Con CelebA solo se usa el
espejo horizontal porque ya tiene mucha variedad.

### Paso 3: fine tuning
Con el clasificador ya entrenado, se descongela **el último bloque
convolucional** (`--bloques 2` descongela los dos últimos) y se entrena todo
con **LR = 1e-5** (100 veces menor que en el paso 2).

* Los últimos bloques tienen rasgos de alto nivel, ajustados a los atributos
  de CelebA; moverlos un poco los especializa en *mi* cara.
* Un LR pequeño evita el **olvido catastrófico** (perder lo aprendido).
* Los primeros bloques (bordes, texturas) se quedan congelados.
* Las **BatchNormalization se dejan congeladas**: con pocos datos, actualizar
  sus medias y varianzas degradaría la red. En Keras, una BN con
  `trainable=False` además trabaja en modo inferencia.
* Después de cambiar `trainable` hay que **volver a compilar**.

### Evaluación: métricas de biometría (`evaluacion.py`)
La accuracy sola no basta para un desbloqueo:

| Métrica | Significado | Tipo de error |
|---|---|---|
| **TAR** (recall) | De mis intentos, cuántos me dejan entrar | — |
| **FRR** = 1 − TAR | Me rechaza a mí | molesto |
| **FAR** | Otra persona logra desbloquear | **peligroso** |

El umbral de 0.5 no es obligatorio: subirlo hace al sistema más estricto (menos
FAR, más FRR). Al final de los pasos 2 y 3 se elige el **umbral más bajo que
logra FAR ≤ 1 %** en validación (`FAR_OBJETIVO`) y se guarda en
`modelos/umbral.json`; `desbloquear.py` lo usa. En validación hay 10 negativos
por cada foto mía para estimar mejor el FAR, y por eso también se reporta la
**accuracy balanceada**.

Gráficas por paso: curvas de entrenamiento (`*_historial.png`), curva ROC
(TAR contra FAR para todos los umbrales), matriz de confusión e histograma de
probabilidades de "yo" contra "otros".

### Extra: réplica del artículo (`extra_cnn_svm_articulo.py`)
Siguiendo la sección 3.1 del artículo:
1. La base del paso 1 se usa **sin re-entrenar** (off-the-shelf).
2. Vector de 256 características → **normalización L2** (norma 1).
3. **SVM lineal** (`LinearSVC`, `class_weight="balanced"`).
4. **CNNaug-SVM**: las fotos de entrenamiento pasan por el aumento aleatorio
   (el *jittering* del artículo: recortes, rotaciones y espejos), y se prueba
   también la transformación de potencia con signo, `sign(x)·|x|^2`, que el
   artículo usa en esa variante.

Si la SVM ya logra buenos resultados, confirma la tesis del artículo: el
trabajo pesado lo hacen las características pre-entrenadas, y el clasificador
de encima puede ser muy simple. Compararla con los pasos 2 y 3 muestra cuánto
aportan el clasificador denso y el fine tuning.

---

## 6. Limitaciones

* Con pocas fotos, la validación tiene pocos positivos: el FAR/FRR estimado
  tiene mucha incertidumbre. Conviene probar `desbloquear.py` con fotos nuevas
  tomadas otro día.
* El umbral se elige sobre el mismo conjunto de validación que se usa para el
  *early stopping*; lo ideal sería un tercer conjunto de prueba.
* Las fotos de CelebA son de celebridades (buena luz, cámaras profesionales) y
  mis fotos son de webcam o celular. La red podría aprender a distinguir
  "tipo de foto" en lugar de "persona". Por eso se recomienda agregar
  `otros_rostros/` tomados con la misma cámara.
* Un sistema real también necesitaría *detección de vida* (anti-spoofing):
  esta red se puede engañar mostrándole una foto impresa.
