"""
Simulacion del desbloqueo: decide si una o varias fotos son de mi rostro.

Carga el modelo final (paso 3; si no existe, el del paso 2) y el umbral
guardado en modelos/umbral.json. Para cada imagen calcula
P(es mi rostro) y la compara con el umbral:

    P >= umbral  ->  DESBLOQUEADO
    P <  umbral  ->  BLOQUEADO

Las fotos deben estar recortadas a la cara (como las de entrenamiento),
porque esta red solo hace la etapa de "extraccion de caracteristicas"; la
deteccion y alineacion del rostro no forman parte del ejercicio.

Uso:
    python desbloquear.py foto1.jpg foto2.png
    python desbloquear.py carpeta_con_fotos/
    python desbloquear.py foto.jpg --umbral 0.8
"""

import argparse
from pathlib import Path

import numpy as np
import tensorflow as tf
import keras

import config
import datos
import evaluacion


def argumentos():
    p = argparse.ArgumentParser(description="Desbloqueo facial")
    p.add_argument("imagenes", nargs="+", help="archivos o carpetas")
    p.add_argument("--umbral", type=float, default=None,
                   help="umbral de decision (por defecto el de umbral.json)")
    p.add_argument("--modelo", type=str, default=None)
    return p.parse_args()


def main():
    args = argumentos()

    if args.modelo:
        ruta_modelo = Path(args.modelo)
    elif config.MODELO_FINAL.exists():
        ruta_modelo = config.MODELO_FINAL
    else:
        ruta_modelo = config.MODELO_ETAPA2
    modelo = keras.models.load_model(ruta_modelo)
    umbral = args.umbral if args.umbral is not None else evaluacion.cargar_umbral()
    print(f"Modelo: {ruta_modelo.name}   umbral: {umbral:.4f}\n")

    rutas = []
    for entrada in args.imagenes:
        rutas += (datos.listar_imagenes(entrada) if Path(entrada).is_dir()
                  else [entrada])

    for ruta in rutas:
        img = datos.cargar_imagen(tf.constant(str(ruta)))
        prob = float(modelo.predict(img[np.newaxis], verbose=0)[0, 0])
        estado = "DESBLOQUEADO" if prob >= umbral else "BLOQUEADO"
        print(f"{estado:13s} P(yo)={prob:.4f}  {ruta}")


if __name__ == "__main__":
    tf.get_logger().setLevel("ERROR")
    main()
