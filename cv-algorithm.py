import cv2
import numpy as np
from pyimagesearch import imutils
from PIL import Image
from time import time

def invert_img(img):
    img = (255-img)
    return img

time_1 = time()

 
#roi = cv2.imread('images/soccer-player-2.jpg')
#roi = cv2.imread('images/surgeon_2.jpg')
#roi = cv2.imread('images/object_group_0.jpg')
roi = cv2.imread('images/beach_trash_3.jpg')

hsv = cv2.cvtColor(roi,cv2.COLOR_BGR2HSV)
 
#target = cv2.imread('images/soccer-player-2.jpg')
#target = cv2.imread('images/surgeon_2.jpg')
#target = cv2.imread('images/object_group_0.jpg')
target = cv2.imread('images/beach_trash_3.jpg')
target = imutils.resize(target, height = 400)

hsvt = cv2.cvtColor(target,cv2.COLOR_BGR2HSV)

img_height = target.shape[0]
img_width = target.shape[1]

# calculating object histogram
roihist = cv2.calcHist([hsv],[0, 1], None, [180, 256], [0, 180, 0, 256] )
 
# normalize histogram and apply backprojection
cv2.normalize(roihist,roihist,0,255,cv2.NORM_MINMAX)
dst = cv2.calcBackProject([hsvt],[0,1],roihist,[0,180,0,256],1)
 
# Now convolute with circular disc
disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
cv2.filter2D(dst,-1,disc,dst)
 
# threshold and binary AND
ret,thresh = cv2.threshold(dst,50,255,0)
thresh_one = thresh.copy()
thresh = cv2.merge((thresh,thresh,thresh))
res = cv2.bitwise_and(target,thresh)
'''
# Showing before morph
thresh_c = thresh_one.copy()
img_c = np.vstack((target,thresh_c,res))
img_c = imutils.resize(img_c, height = 700)
cv2.imshow('Before morph', thresh_c)
'''

# Implementing morphological erosion & dilation
kernel = np.ones((9,9),np.uint8)  # (6,6) to get more contours (9,9) to reduce noise
thresh_one = cv2.erode(thresh_one, kernel, iterations = 3)
thresh_one = cv2.dilate(thresh_one, kernel, iterations=2)


# Invert the image
thresh_one = invert_img(thresh_one)

# Preforming grab-cut
'''
mask = np.zeros(target.shape[:2],np.uint8)
bgdModel = np.zeros((1,65),np.float64)
fgdModel = np.zeros((1,65),np.float64)
rect = (0,0,img_height,img_width)
cv2.grabCut(thresh_one,mask,rect,bgdModel,fgdModel,5,cv2.GC_INIT_WITH_RECT)
'''

# To show prev img

#res = np.vstack((target,thresh,res))
#cv2.imwrite('res.jpg',res)

#cv2.waitKey(0)


#cv2.imshow('Before contours', thresh_one)
cnt_target = target.copy()
# Code to draw the contours
contours, hierarchy = cv2.findContours(thresh_one.copy(),cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
cnts = sorted(contours, key = cv2.contourArea, reverse = True)

cv2.drawContours(cnt_target, cnts, -1,(0,0,255),2)
print time() - time_1

res = imutils.resize(thresh_one, height = 700)
cv2.imshow('Original image', target)
cv2.imshow('Preprocessed', thresh_one)
cv2.imshow('All contours', cnt_target)

cv2.waitKey(0)



