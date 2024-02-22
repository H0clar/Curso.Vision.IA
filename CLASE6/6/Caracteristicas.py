import numpy as np
import cv2

cap = cv2.VideoCapture(0)

im1 = cv2.imread('frontal.png')
ancho = int(im1.shape[1]/5)
alto = int(im1.shape[0]/5)
im1 = cv2.resize(im1, (ancho, alto), interpolation = cv2.INTER_AREA)


while True:
    ret, frame = cap.read()


    #escala de grises

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_im1 = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)


    #puntos claves
    #numero de puntos

    num_kpt = 500
    #detectar el objeto

    orb = cv2.ORB_create(num_kpt)

    #extraer informacion de la imagen

    keypoint1, descriptor1 = orb.detectAndCompute(gray_im1, None)

    #extraer informacion de los frames

    keypoint2, descriptor2 = orb.detectAndCompute(gray_frame, None)


    #dibujamos

    im1_display = cv2.drawKeypoints(im1, keypoint1, outImage=np.array([]), color=(255, 0, 0), flags=cv2.DRAW_MATCHES_FLAGS_DEFAULT)

    frame_display = cv2.drawKeypoints(frame, keypoint2, outImage=np.array([]), color=(255, 0, 0), flags=cv2.DRAW_MATCHES_FLAGS_DEFAULT)


    cv2.imshow("videocaptura", frame_display)
    cv2.imshow("imagen", im1_display)


    #cerrar al precionar escape

    if cv2.waitKey(1) == 27:
        break

cap.release()



