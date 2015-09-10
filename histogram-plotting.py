import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('images/beach_trash_3.jpg')
color = ('b','g','r')

for i, col in enumerate(color):
    histr = cv2.calcHist([img],[i],None,[256],[0,256])
    plt.plot(histr, color = col)
    plt.xlim([0,256])

cv2.imshow('Original image', img)
plt.show()

img_gray = cv2.imread('images/beach_trash_3.jpg',0)
cv2.imshow('Original image', img_gray)
plt.hist(img_gray.ravel(),256,[0,256])
plt.show()


