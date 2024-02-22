import cv2
import numpy as np
import matplotlib.pyplot as plt

#leer la imagen

img = cv2.imread('images/monedas.png')

imgmat = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


#convertir la imagen a escala de grises

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


#definir la matriz

matriz = np.ones(gray.shape, dtype='uint8')*50
matrizrgb = np.ones(img.shape, dtype='uint8')*60


#Aumentar el brillo

brillantergb = cv2.add(img, matrizrgb)
brillantergbm = cv2.cvtColor(brillantergb, cv2.COLOR_BGR2RGB)

#disminuir el brillo

oscurorgb = cv2.subtract(img, matrizrgb)
oscurorgbm = cv2.cvtColor(oscurorgb, cv2.COLOR_BGR2RGB)

#aumentar el brillo de las imagenes en grises

brillantegray = cv2.add(gray, matriz)
#disminuir el brillo de las imagenes en grises
oscurogray = cv2.subtract(gray, matriz)

#mostrar imagenes

fig = plt.figure()

#imagen original

ax1 = fig.add_subplot(2, 3, 1)
ax1.imshow(imgmat)
ax1.set_title('imagen RGB')

#BRILLANTE RGB

ax3 = fig.add_subplot(2, 3, 2)
ax3.imshow(brillantergbm)
ax3.set_title('imagen brillante RGB')


#OSCURA RGB

ax4 = fig.add_subplot(2, 3, 3)
ax4.imshow(oscurorgbm)
ax4.set_title('imagen oscura RGB')




#imagen gris

ax2 = fig.add_subplot(2, 3, 4)
ax2.imshow(gray, cmap='gray')
ax2.set_title('imagen gris')

#BRILLANTE GRIS

ax3 = fig.add_subplot(2, 3, 5)
ax3.imshow(brillantegray, cmap='gray')
ax3.set_title('imagen brillante GRIS')

#oscura gris

ax4 = fig.add_subplot(2, 3, 6)
ax4.imshow(oscurogray, cmap='gray')
ax4.set_title('imagen oscura GRIS')

plt.show()








