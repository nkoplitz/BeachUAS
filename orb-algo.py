import numpy as np
from pyimagesearch import imutilspy
import cv2
from matplotlib import pyplot as plt

#img = cv2.imread('dataset/data/water_bottle/2.jpg')
img = cv2.imread('dataset/data/water_bottle/21.jpg')
img = imutilspy.resize(img, height = 400)

# Image processing
#img = cv2.GaussianBlur(img, (5, 5), 0)
img = cv2.medianBlur(img,3)
#img = cv2.Canny(img,75,500)

# Initiate STAR detector
orb = cv2.ORB_create()

# find the keypoints with ORB
kp = orb.detect(img,None)

# compute the descriptors with ORB
kp, des = orb.compute(img, kp)

# draw only keypoints location,not size and orientation
img2 = cv2.drawKeypoints(img,kp,img,color=(0,255,0), flags=0)
plt.imshow(img),plt.show()
