from turtle import width
import dlib
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

img = cv.imread("Andres_Sanchez.jpg")
b, g, r = cv.split(img)
height, width, channels = img.shape


# plt.hist(b.ravel(), 256, [0, 256],color='b')
# plt.hist(g.ravel(), 256, [0, 256],color='g')
# plt.hist(r.ravel(), 256, [0, 256],color='r')
# plt.show()

img_eq_b = dlib.equalize_histogram(b)
img_eq_g = dlib.equalize_histogram(g)
img_eq_r = dlib.equalize_histogram(r)

# plt.hist(img_eq_b.ravel(), 256, [0, 256],color='b')
# plt.show()
# plt.hist(img_eq_g.ravel(), 256, [0, 256],color='g')
# plt.show()
# plt.hist(img_eq_r.ravel(), 256, [0, 256],color='r')
# plt.show()

img_eq = np.zeros_like(img)
img_eq[:,:,0] = img_eq_b
img_eq[:,:,1] = img_eq_g
img_eq[:,:,2] = img_eq_r
print(img_eq.shape)
cv.imshow("image",img)
cv.imshow("image_eq",img_eq)
# cv.imshow("image_b",img_eq_b)
# cv.imshow("image_g",img_eq_g)
# cv.imshow("image_r",img_eq_r)
# cv.imshow("b", b)
# cv.imshow("g", g)
# cv.imshow("r", r)
cv.waitKey()
cv.destroyAllWindows()
