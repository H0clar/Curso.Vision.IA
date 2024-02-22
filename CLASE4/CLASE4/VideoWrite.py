import cv2


cap = cv2.VideoCapture(0)
ancho = int (cap.get(cv2.CAP_PROP_FRAME_WIDTH))
alto = int (cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

print(ancho, alto)
#formato y fps del video de salida
out = cv2.VideoWriter('videoSalida.mp4', cv2.VideoWriter_fourcc(*'XVID'), 60, (ancho, alto))


while True:
    ret, frame = cap.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    out.write(frame)

    cv2.imshow("video rgb", frame)

    #cerrar con escape

    if cv2.waitKey(1) == 27:
        break


cap.release()
out.release()
cv2.destroyAllWindows()


