import cv2
import dlib
import get_points

cap = cv2.VideoCapture(0)

print("Pulsa 'p' para pausar el video y empezar el seguimiento")

def tracker(image, puntos):
    tracker = dlib.correlation_tracker()
    tracker.start_track(image, dlib.rectangle(*puntos[0]))

    while True:
        ret, image = cap.read()
        if not ret:
            print("No se ejecutó la captura")
            exit()

        tracker.update(image)

        rect = tracker.get_position()
        pt1 = (int(rect.left()), int(rect.top()))
        pt2 = (int(rect.right()), int(rect.bottom()))
        cv2.rectangle(image, pt1, pt2, (255, 255, 255), 3)
        print("Objeto trackeado en [{},{}] \r".format(pt1, pt2), )
        loc = (int(rect.left()), int(rect.top() - 20))
        texto = "Objeto trackeado en [{},{}]".format(pt1, pt2)
        cv2.putText(image, texto, loc, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.imshow("imagen", image)

        # Cerrar con escape
        if cv2.waitKey(1) == 27:
            break

while True:
    ret, frame = cap.read()

    t = cv2.waitKey(1)

    if not ret:
        print("No se ejecutó la captura")
        exit()

    if t == ord('p'):
        points = get_points.run(frame)
        if not points:
            print("No se seleccionó ningún objeto")
            exit()
        if points:
            tracker(image=frame, puntos=points)
        break

    cv2.imshow("imagen", frame)

cap.release()
cv2.destroyAllWindows()
