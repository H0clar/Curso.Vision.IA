import cv2
import numpy as np
import matplotlib.pyplot as plt



#buscar imagen
img = cv2.imread('images/monedas.png')

#convertir imagen a escala de grises

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#definir la matriz

matriz = np.ones(gray.shape, dtype='uint8')*50

#Aumentar el brillo en grises

brillantegray = cv2.add(gray, matriz)

#threshold

_, imgthreshold = cv2.threshold(brillantegray, 160, 255, cv2.THRESH_BINARY)
_, imgthreshold2 = cv2.threshold(brillantegray, 160, 255, cv2.THRESH_BINARY_INV)



#adaptive

imgadaptative = cv2.adaptiveThreshold(brillantegray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)




#disminuir el brillo en grises

oscuragray = cv2.subtract(gray, matriz)

_, imgthreshold3 = cv2.threshold(oscuragray, 50, 255, cv2.THRESH_BINARY)
_, imgthreshold4 = cv2.threshold(oscuragray, 50, 255, cv2.THRESH_BINARY_INV)

#adaptive

imgadaptive2 = cv2.adaptiveThreshold(oscuragray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 7)


#MOSTRAR LA IMAGEN

fig = plt.figure()

#brillante

ax1 = fig.add_subplot(2, 4, 1)
ax1.imshow(brillantegray, cmap='gray')
ax1.set_title('imagen brillante')


#brillante threshold 1

ax3 = fig.add_subplot(2, 4, 2)
ax3.imshow(imgthreshold, cmap='gray')
ax3.set_title('imagen threshold 1')

#brillante threshold 2

ax4 = fig.add_subplot(2, 4, 3)
ax4.imshow(imgthreshold2, cmap='gray')
ax4.set_title('imagen threshold 2')

#brillante adaptive

ax5 = fig.add_subplot(2, 4, 4)
ax5.imshow(imgadaptative, cmap='gray')
ax5.set_title('imagen brillante adaptive')

#oscura

ax2 = fig.add_subplot(2, 4, 5)
ax2.imshow(oscuragray, cmap='gray')
ax2.set_title('imagen oscura')

#oscura threshold 1

ax3 = fig.add_subplot(2, 4, 6)
ax3.imshow(imgthreshold3, cmap='gray')
ax3.set_title('imagen threshold 1')

#oscura threshold 2

ax4 = fig.add_subplot(2, 4, 7)
ax4.imshow(imgthreshold4, cmap='gray')
ax4.set_title('imagen threshold 2')

#oscura adaptive

ax5 = fig.add_subplot(2, 4, 8)
ax5.imshow(imgadaptive2, cmap='gray')
ax5.set_title('imagen oscura adaptive')

plt.show()










