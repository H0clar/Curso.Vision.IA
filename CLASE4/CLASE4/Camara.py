import cv2
import os

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    cv2.imshow('VENTANA DE IMAGEN', frame)


    #cerrar con la tecla esc


    if cv2.waitKey(1) == 27:
        break


cap.release()

cv2.destroyAllWindows()


