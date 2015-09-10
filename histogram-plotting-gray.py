import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('images/beach_trash_3.jpg',0)
cv2.imshow('Original image', img)
plt.hist(img.ravel(),256,[0,256]); plt.show()
