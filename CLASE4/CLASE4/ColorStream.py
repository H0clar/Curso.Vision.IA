import cv2


cap = cv2. VideoCapture(0)

while True:
    ret, frame = cap.read()
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    edg = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


    print(ret)

    cv2.imshow("video rgb", frame)
    cv2.imshow("video hsv", hsv)
    cv2.imshow("video edg", edg)

    #cerrar con escape

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
