import cv2
import numpy as np

# Constantes para teclas
TECLA_ESC = 27
TECLA_NORMAL = 48
TECLA_DESENFOQUE = 49
TECLA_ESQUINAS = 50
TECLA_BORDES = 51
TECLA_ROSTROS = 52
TECLA_PUNTOS_FACIALES = 53
TECLA_REINICIAR = 48  # Nueva tecla para reiniciar

# Parámetros para detección de esquinas
esquinas_param = dict(maxCorners=500, qualityLevel=0.2, minDistance=15, blockSize=9)

# Captura de video
cap = cv2.VideoCapture(0)

# Inicializar el estado de ánimo (mood) y el diccionario de estados de teclas
mood = TECLA_NORMAL
tecla_actual = None
tecla_anterior = None

# Clasificadores para detección de rostros y puntos faciales
rostro_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
puntos_faciales_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')

# Documentación del script
"""
Este script utiliza OpenCV para aplicar diferentes efectos de procesamiento de imagen en tiempo real.
Puedes cambiar el estado de ánimo (mood) presionando y manteniendo presionadas diferentes teclas.
- 48: Normal
- 49: Desenfoque
- 50: Esquinas
- 51: Bordes
- 52: Detección de rostros

- 48: Reiniciar efecto
Presiona la tecla 'Esc' para salir.
"""

while True:
    ret, frame = cap.read()

    # Almacena la tecla anterior antes de actualizarla
    tecla_anterior = tecla_actual

    # Verificar si alguna tecla está siendo presionada
    k = cv2.waitKey(1) & 0xFF
    if k != 255:
        tecla_actual = k

    # Aplicar efectos según el estado de ánimo (mood)
    if tecla_actual == TECLA_DESENFOQUE:
        resultado = cv2.blur(frame, (13, 13))
    elif tecla_actual == TECLA_BORDES:
        resultado = cv2.Canny(frame, 135, 150)
    elif tecla_actual == TECLA_ESQUINAS:
        resultado = frame
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        esquinas = cv2.goodFeaturesToTrack(gray, **esquinas_param)
        if esquinas is not None:
            for x, y in np.float32(esquinas).reshape(-1, 2):
                x, y = int(x), int(y)
                cv2.circle(resultado, (x, y), 5, (0, 255, 0), -1)
    elif tecla_actual == TECLA_ROSTROS:
        resultado = frame
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rostros = rostro_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))
        for (x, y, w, h) in rostros:
            cv2.rectangle(resultado, (x, y), (x+w, y+h), (255, 0, 0), 2)

    elif tecla_actual == TECLA_REINICIAR and tecla_anterior != TECLA_REINICIAR:
        resultado = frame
        tecla_actual = None  # Reiniciar efecto

    else:
        resultado = frame

    # Mostrar resultado
    cv2.imshow('resultado', resultado)

    # Salir con la tecla 'Esc'
    if k == TECLA_ESC:
        break

# Liberar recursos
cap.release()
cv2.destroyAllWindows()
