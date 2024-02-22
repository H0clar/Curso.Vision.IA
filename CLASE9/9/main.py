import cv2

model = 'frozen_inference_graph.pb'
config = 'ssd_mobilenet_v2_coco_2018_03_29.pbtxt'
clases = 'COCO_labels.txt'

with open(clases) as cl:
    labels = cl.read().split('\n')
print(labels)

net = cv2.dnn.readNetFromTensorflow(model, config)

def object_detect(net, img):
    dim = (300)

    blob = cv2.dnn.blobFromImage(img, 1.0, size=(dim, dim), mean=(0, 0, 0), swapRB=True, crop=False)

    net.setInput(blob)

    objetos = net.forward()

    return objetos

def text(img, label, x, y):
    sizetext = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 1)

    dim = sizetext[0]
    baseline = sizetext[1]

    # rectangulos:
    cv2.rectangle(img, (x, y - dim[1] - baseline), (x + dim[0], y + baseline), (0, 0, 0), cv2.FILLED)

    # texto:
    cv2.putText(img, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 1)

def dibujar_objetos(img, objetos, umbral=0.5):
    filas = img.shape[0]
    colum = img.shape[1]

    for i in range(objetos.shape[2]):
        clase = int(objetos[0, 0, i, 1])
        puntaje = float(objetos[0, 0, i, 2])

        x = int(objetos[0, 0, i, 3] * colum)
        y = int(objetos[0, 0, i, 4] * filas)
        w = int(objetos[0, 0, i, 5] * colum - x)
        h = int(objetos[0, 0, i, 6] * filas - y)

        if puntaje > umbral:
            text(img, "{}: {:.2f}%".format(labels[clase], puntaje * 100), x, y)
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

cap = cv2.VideoCapture(0)

cap.set(3, 1090)
cap.set(4, 1920)

while True:
    ret, frame = cap.read()
    detect = object_detect(net, frame)

    dibujar_objetos(frame, detect)

    cv2.imshow('VIDEO CAPTURA', frame)

    # CERRAR CON ESCAPE
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
