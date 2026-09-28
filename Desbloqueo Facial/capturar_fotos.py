"""
(Opcional) Captura fotos de tu rostro con la webcam y las recorta a la cara.

La tarea supone que la red recibe fotos que YA son rostros (las etapas de
deteccion y alineacion no se piden). Este script resuelve esas dos etapas de
forma sencilla para construir datos/mi_rostro/:

  * Deteccion: clasificador en cascada de Haar incluido en OpenCV.
  * "Alineacion": se recorta un cuadrado alrededor de la cara con un margen,
    parecido al encuadre de las imagenes alineadas de CelebA.

Requiere OpenCV:  pip install opencv-python

Uso:
    python capturar_fotos.py                 # guarda en datos/mi_rostro/
    python capturar_fotos.py --destino datos/otros_rostros --prefijo amigo
    python capturar_fotos.py --recortar-carpeta fotos_del_celular/

Controles en la ventana: ESPACIO = guardar foto, A = modo automatico
(una foto cada ~0.5 s), Q / ESC = salir. Muevete, cambia de expresion,
ponte y quitate lentes y cambia la iluminacion: mas variedad = mejor modelo.
"""

import argparse
import time
from pathlib import Path

import config
import datos

try:
    import cv2
except ImportError as e:
    raise SystemExit("Este script necesita OpenCV: pip install opencv-python") from e


MARGEN = 0.35   # margen extra alrededor de la cara detectada (35%)


def detector():
    return cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml")


def recortar_cara(imagen, cascada):
    """Detecta la cara mas grande y regresa un recorte cuadrado (o None)."""
    gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    caras = cascada.detectMultiScale(gris, scaleFactor=1.1, minNeighbors=5,
                                     minSize=(80, 80))
    if len(caras) == 0:
        return None, None
    x, y, w, h = max(caras, key=lambda c: c[2] * c[3])
    lado = int(max(w, h) * (1 + 2 * MARGEN))
    cx, cy = x + w // 2, y + h // 2
    x0, y0 = max(cx - lado // 2, 0), max(cy - lado // 2, 0)
    x1 = min(x0 + lado, imagen.shape[1])
    y1 = min(y0 + lado, imagen.shape[0])
    return imagen[y0:y1, x0:x1], (x, y, w, h)


def siguiente_nombre(destino, prefijo):
    existentes = len(list(destino.glob(f"{prefijo}_*.jpg")))
    return destino / f"{prefijo}_{existentes + 1:04d}.jpg"


def recortar_carpeta(carpeta, destino, prefijo, cascada):
    """Recorta las caras de fotos ya existentes (por ejemplo del celular)."""
    guardadas = 0
    for ruta in datos.listar_imagenes(carpeta):
        imagen = cv2.imread(ruta)
        if imagen is None:
            continue
        cara, _ = recortar_cara(imagen, cascada)
        if cara is None:
            print(f"  sin cara detectada: {ruta}")
            continue
        cv2.imwrite(str(siguiente_nombre(destino, prefijo)), cara)
        guardadas += 1
    print(f"Guardadas {guardadas} caras en {destino}")


def capturar_webcam(destino, prefijo, cascada, camara):
    cap = cv2.VideoCapture(camara)
    if not cap.isOpened():
        raise SystemExit("No se pudo abrir la camara")
    automatico, ultimo, guardadas = False, 0.0, 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        cara, caja = recortar_cara(frame, cascada)
        vista = frame.copy()
        if caja is not None:
            x, y, w, h = caja
            cv2.rectangle(vista, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(vista, f"guardadas: {guardadas}  auto: {automatico}",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.imshow("Captura (ESPACIO guardar, A auto, Q salir)", vista)

        tecla = cv2.waitKey(1) & 0xFF
        if tecla in (ord("q"), 27):
            break
        if tecla == ord("a"):
            automatico = not automatico
        guardar = tecla == ord(" ") or (automatico and time.time() - ultimo > 0.5)
        if guardar and cara is not None:
            cv2.imwrite(str(siguiente_nombre(destino, prefijo)), cara)
            guardadas += 1
            ultimo = time.time()
    cap.release()
    cv2.destroyAllWindows()
    print(f"Guardadas {guardadas} fotos en {destino}")


def main():
    p = argparse.ArgumentParser(description="Captura de fotos del rostro")
    p.add_argument("--destino", default=str(config.MI_ROSTRO_DIR))
    p.add_argument("--prefijo", default="yo")
    p.add_argument("--camara", type=int, default=0)
    p.add_argument("--recortar-carpeta", default=None,
                   help="en vez de la webcam, recorta caras de esta carpeta")
    args = p.parse_args()

    destino = Path(args.destino)
    destino.mkdir(parents=True, exist_ok=True)
    cascada = detector()
    if args.recortar_carpeta:
        recortar_carpeta(args.recortar_carpeta, destino, args.prefijo, cascada)
    else:
        capturar_webcam(destino, args.prefijo, cascada, args.camara)


if __name__ == "__main__":
    main()
