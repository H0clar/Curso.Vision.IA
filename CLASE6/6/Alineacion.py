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


    num_kpt = 500

    orb = cv2.ORB_create(num_kpt)
    keypoint1, descriptor1 = orb.detectAndComputer(gray_im1, None)

    keypoint2, descriptor2 = orb.detectAndCompute(gray_frame, None)



    #dibujamos

    im1_display = cv2.drawKeypoints(im1, keypoint1, outImage=np.array([]), color=(255, 0, 0), flags=cv2.DRAW_MATCHES_FLAGS_DEFAULT)
    frame_display = cv2.drawKeypoints(frame, keypoint2, outImage=np.array([]), color=(255, 0, 0), flags=cv2.DRAW_MATCHES_FLAGS_DEFAULT)



    #comparar los puntos

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = matcher.match(descriptor1, descriptor2)

    #ordenar la lista de puntos

    matches = sorted(matches, key=lambda x:x.distance, reverse=False)

    #filtrar los resultados

    goodmatches = int(len(matches)*0.1)
    matches = matches[:goodmatches]

    #mostrar coincidencias

    img_matches = cv2.drawMatches(im1, keypoint1, frame, keypoint2, matches, None)



    #calcular homografia de la imagen

    puntos1 = np.zeros((len(matches), 2), dtype=np.float32)
    puntos2 = np.zeros((len(matches),2), dtype=np.float32)

    #extraer puntos de las coincidencias

    for i, match in enumerate(matches):
        puntos1[i,:] = keypoint1[match.queryIdx].pt
        puntos2[i,:] = keypoint2[match.trainIdx].pt

    #calcular homografia

    h, mask = cv2.findHomography(puntos2, puntos1, cv2.RANSAC)

    #dibujamos el cuadro

    alto, ancho, canales = im1.shape
    img_perspec = cv2.warpPerspective(frame, h, (ancho, alto))



    #mostrar caracteristicas

    cv2.imshow("video captura", frame_display)
    cv2.imshow("imagen", im1_display)
    cv2.imshow("coincidencias", img_perspec)

    #cerrar al precionar escape

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()

