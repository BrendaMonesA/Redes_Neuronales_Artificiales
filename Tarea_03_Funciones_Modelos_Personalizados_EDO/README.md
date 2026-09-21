# Tarea 3: Funciones, Modelos personalizados y Ecuaciones Diferenciales

## Problemas

1. **Capa RGB → escala de grises.** Diseñar una capa en Keras que transforme
   imágenes a color en escala de grises. Solo se pide el diseño de la capa (no
   hay nada que entrenar). Se puede usar MNIST u otra base para las pruebas.

2. **Aproximación de funciones.** Entrenar una red neuronal para que reproduzca
   las siguientes funciones en el intervalo `[-1, 1]` y graficar la solución de
   la red junto con la función:
   - (a) `3 sin(pi x)`
   - (b) `1 + 2x + 4x^3`

3. **Capa entrenable polinomial (grado 3).** Diseñar una capa entrenable que
   represente `f(x) = a0 + a1 x + a2 x^2 + a3 x^3`, con `a0, a1, a2, a3` como
   parámetros entrenables. Entrenarla para ajustar `f(x) = cos(2x)` en `[-1, 1]`.

4. **Ecuaciones diferenciales (PINN).** Entrenar una red neuronal que dé la
   solución de las siguientes ecuaciones diferenciales en el intervalo `[-5, 5]`.
   Graficar la solución numérica junto con la solución analítica:
   - (a) `x y' + y = x^2 cos(x)` con `y(0) = 0`
   - (b) `d^2y/dx^2 = -y` con `y(0) = 1`, `y'(0) = -0.5`

## Nota
Subir la tarea mostrando claramente el código y los resultados/gráficas de cada
problema e inciso.
