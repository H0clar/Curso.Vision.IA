import cv2


cap = cv2.VideoCapture("videoSalida.mp4")

while True:
    ret, frame = cap.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    cv2.imshow("video rgb", frame)

    #cerrar con escape

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()



