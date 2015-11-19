import numpy as np
from pyimagesearch import imutilspy
import cv2
from matplotlib import pyplot as plt

def grabcut(img):
    mask = np.zeros(img.shape[:2],np.uint8)
    bgdModel = np.zeros((1,65),np.float64)
    fgdModel = np.zeros((1,65),np.float64)
    #rect = (50,50,450,290)
    rect = (0,0,img.shape[0],img.shape[1])

    cv2.grabCut(img,mask,rect,bgdModel,fgdModel,1,cv2.GC_INIT_WITH_RECT)
    mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
    img = img*mask2[:,:,np.newaxis]
    return img
def invert_img(img):
    img = (255-img)
    return img

def histBackProj(img):
    hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    hsvt = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)

    img_height = img.shape[0]
    img_width = img.shape[1]

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
    return thresh

#img = cv2.imread('dataset/data/water_bottle/2.jpg')
#img = cv2.imread('dataset/data/water_bottle/21.jpg')
#img = cv2.imread('dataset/data_two_choice/test/paper/1.jpg')
#img = cv2.imread('dataset/data_two/paper/IMG_0205.JPG',0)
#img = cv2.imread('dataset/data_two/water_bottle/IMG_0187.JPG',0)
#img = cv2.imread('dataset/data_two/water_bottle/IMG_0155.JPG')
#img = cv2.imread('dataset/data_two/water_bottle/IMG_0171.JPG')
img = cv2.imread('dataset/data_two/water_bottle/IMG_0163.JPG')

img = imutilspy.resize(img, height = 400)



img = histBackProj(img)
img = invert_img(img)



cv2.imshow('test', img)
cv2.waitKey(0)
'''
img = cv2.medianBlur(img,5)
ret,img = cv2.threshold(img,127,255,cv2.THRESH_BINARY)

'''

#img = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
#img = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,11,2)
#img = grabcut(img)




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
#plt.imshow(img),plt.show()
cv2.imshow('Result',img2)
cv2.waitKey(0)

