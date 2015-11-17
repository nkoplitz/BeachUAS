import numpy as np
from pyimagesearch import imutilspy
import cv2
from matplotlib import pyplot as plt

#img = cv2.imread('dataset/data/water_bottle/2.jpg')
#img = cv2.imread('dataset/data/water_bottle/21.jpg')
#img = cv2.imread('dataset/data_two_choice/test/paper/1.jpg')
img = cv2.imread('dataset/data_two/paper/IMG_0205.JPG')
#img = cv2.imread('dataset/data_two/water_bottle/IMG_0188.JPG')
#img = cv2.imread('dataset/data_two/water_bottle/IMG_0166.JPG')

img = imutilspy.resize(img, height = 400)


# Grab cut experimentation #
mask = np.zeros(img.shape[:2],np.uint8)
bgdModel = np.zeros((1,65),np.float64)
fgdModel = np.zeros((1,65),np.float64)
#rect = (50,50,450,290)
rect = (0,0,img.shape[0],img.shape[1])

cv2.grabCut(img,mask,rect,bgdModel,fgdModel,1,cv2.GC_INIT_WITH_RECT)
mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
img = img*mask2[:,:,np.newaxis]
############################

# Image processing
#img = cv2.GaussianBlur(img, (5, 5), 0)
#img = cv2.medianBlur(img,3)
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
