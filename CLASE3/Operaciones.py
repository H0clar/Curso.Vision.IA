import cv2
import numpy as np
import matplotlib.pyplot as plt

from Umbralizacion import fig

#imagenes

img1 = cv2.imread('images/img1.png')
img2 = cv2.imread('images/img2.png')

#operacion and
imgrand = cv2.bitwise_and(img1, img2, mask=None)

#operacion or
imgor = cv2.bitwise_or(img1, img2, mask=None)

#operacion xor
imgxor = cv2.bitwise_xor(img1, img2, mask=None)

#mostrar imagenes

fog = plt.figure()

#imagen 1

ax1 = fig.add_subplot(2, 3, 1)
ax1.imshow(img1, cmap='gray')
ax1.set_title('imagen 1')

#imagen 2

ax2 = fig.add_subplot(2, 3, 4)
ax2.imshow(img2, cmap='gray')
ax2.set_title('imagen 2')


#imagen and
ax3 = fig.add_subplot(2, 3, 2)
ax3.imshow(imgrand, cmap='gray')
ax3.set_title('imagen and')


#imagen or

ax4 = fig.add_subplot(2, 3, 5)
ax4.imshow(imgrand, cmap='gray')
ax4.set_title('imagen or')

#imagen xor

ax5 = fig.add_subplot(2, 3, 5)
ax5.imshow(imgxor, cmap='gray')
ax5.set_title('imagen xor')

#aplicacion con imagen

imglogo = cv2.imread('images/logo.png')
imgback = cv2.imread('images/fondo.png')

#correccion de color

imglogo = cv2.cvtColor(imglogo, cv2.COLOR_BGR2RGB)
imgback = cv2.cvtColor(imgback, cv2.COLLOR_BGR2RGB)

#MASCARA

logogray = cv2.cvtColor(imglogo, cv2.COLOR_BGR2GRAY)
_, imgmask = cv2.threshold(logogray, 127, 255, cv2.THRESH_BINARY)

#invertir la mascara

imgmask_inv = cv2.bitwise_not(imgmask)

#realizar operaciones

imgapp1 = cv2.bitwise_and(imgback, imgback, mask=imgmask_inv)

imgapp2 = cv2.bitwise_and(imglogo, imglogo, mask=imgmask)

imgapp3 = cv2.subtract(imgback, imgapp2)

#mostrar imagenes

fig1 = plt.figure()


#imagen 1

ax11 = fig1.add_subplot(3, 3, 1)
ax11.imshow(imglogo)
ax11.set_title('imagen 1')

#imagen 2

ax22 = fig1.add_subplot(3, 3, 4)
ax22.imshow(imgback)
ax22.set_title('imagen 2')

#img mask

ax33 = fig1.add_subplot(3, 3, 2)
ax33.imshow(imgmask, cmap='gray')
ax33.set_title('imagen mask')

#mascara invertida

ax44 = fig1.add_subplot(3, 3, 3)
ax44.imshow(imgmask_inv, cmap='gray')
ax44.set_title('imagen mask invertida')

#imagen and

ax44 = fig1.add_subplot(3, 3, 5)
ax44.imshow(imgapp1)
ax44.set_title('imagen and')

#imagen and 2

ax44 = fig1.add_subplot(3, 3, 6)
ax44.imshow(imgapp2)
ax44.set_title('imagen and 2')

#imagen add

ax44 = fig1.add_subplot(3, 3, 7)
ax44.imshow(imgapp3)
ax44.set_title('imagen add')

plt.show()




