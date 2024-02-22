import cv2

model = 'pose_deploy_linevec_faster_4_stages.prototxt'
pesos = 'pose_iter_160000.caffemodel'

numpuntos = 15

pares = [[0, 1], [1, 2], [2, 3], [3, 4], [1, 5], [5, 6], [6, 7], [1, 14], [14, 8], [8, 9], [9, 10], [14, 11], [11, 12], [12, 13]]

net = cv2.dnn.readNetFromCaffe(model, pesos)

cap = cv2.VideoCapture(0)

p = False
e = False

while True:
    ret, frame = cap.read()

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    anchoframe = frame.shape[1]
    altoframe = frame.shape[0]

    TamEntNet = (368, 368)
    blob = cv2.dnn.blobFromImage(frame, 1.0 / 255, TamEntNet, (0, 0, 0), swapRB=False, crop=False)

    net.setInput(blob)
    output = net.forward()

    scalex = anchoframe / output.shape[3]
    scaley = altoframe / output.shape[2]

    puntos = []
    umbral = 0.1

    for i, probmap in enumerate(output[0, :numpuntos, :, :]):
        minVal, prob, minLoc, point = cv2.minMaxLoc(probmap)

        x = scalex * point[0]
        y = scaley * point[1]

        if prob > umbral:
            puntos.append((int(x), int(y)))
        else:
            puntos.append(None)

    key = cv2.waitKey(1) & 0xFF

    if key == 27:  # Escape
        break
    elif key == ord('p'):
        p = not p
        e = False
    elif key == ord('e'):
        e = not e
        p = False

    if p:
        for i, punto in enumerate(puntos):
            if punto is not None:
                x, y = punto
                cv2.circle(frame, (int(x), int(y)), 8, (255, 0, 255), thickness=-1, lineType=cv2.FILLED)
                cv2.putText(frame, "{}".format(i), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, lineType=cv2.LINE_AA)

    elif e:
        for par in pares:
            parteA, parteB = par
            if puntos[parteA] and puntos[parteB]:
                cv2.line(frame, puntos[parteA], puntos[parteB], (0, 255, 255), 2)

    cv2.imshow('frame', frame)

cap.release()
cv2.destroyAllWindows()
