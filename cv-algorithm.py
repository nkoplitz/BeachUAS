import cv2
import numpy as np
from pyimagesearch import imutilspy
from PIL import Image
from time import time

def invert_img(img):
    img = (255-img)
    return img

def threshold(im):
    imgray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    imgray = cv2.medianBlur(imgray,9)
    imgray = cv2.Canny(imgray,75,200)
    
    return imgray

def view_all_contours(im, size_min, size_max, visual):
    main = np.array([[]])
    cnt_target = im.copy()
    
    for c in cnts:
        epsilon = 0.1*cv2.arcLength(c,True)
        approx = cv2.approxPolyDP(c,epsilon,True)
        area = cv2.contourArea(c)
        print 'area: ', area
        test = im.copy()
        
        
        
        
        #print 'Contours: ', contours
        # To weed out contours that are too small
        if area > size_min and area < size_max:
            print c[0,0]
            print 'approx: ', len(approx)
            #print 'epsilon: ', epsilon
            max_pos = c.max(axis=0)
            max_x = max_pos[0,0]
            max_y = max_pos[0,1]

            min_pos = c.min(axis=0)
            min_x = min_pos[0,0]
            min_y = min_pos[0,1]
            
            
            
            # Load each contour onto image
            cv2.drawContours(cnt_target, c, -1,(0,0,255),2)
            #cv2.drawContours(test, c, -1,(0,0,255),2)
            #cv2.imshow('Original image', cnt_target)
            
            
            print 'Found object'
            #print 'Approx.shape: ', approx.shape
            #print 'Test.shape: ', test.shape
            
            #frame_f = frame_f[y: y+h, x: x+w]
            frame_f = test[min_y:max_y , min_x:max_x]

            #print 'frame_f.shape: ', frame_f.shape
            main = np.append(main, approx[None,:][None,:])
            #print 'main: ', main

            thresh = frame_f.copy()
            thresh = threshold(thresh)

            __ , contours_small, hierarchy = cv2.findContours(thresh.copy(),cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
            cnts_small = sorted(contours_small, key = cv2.contourArea, reverse = True)
            cv2.drawContours(frame_f, cnts_small, -1,(0,0,255),2)
            if visual:
                cv2.imshow('Thresh', thresh)
                cv2.imshow('Show Ya', frame_f)
                cv2.waitKey(0)
            

            
        #cv2.imshow('Show Ya', test)
        #print 'Approx: ', approx.shape

        # Uncomment in order to show all rectangles in image
        
        
    print '---------------------------------------------'
    #cv2.drawContours(cnt_target, cnts, -1,(0,255,0),2)
    print main.shape
    print main
    return cnt_target



time_1 = time()

#path = 'images/surgeon_2.jpg'
#path = 'images/beach_trash_3.jpg'
path = 'images/beach_trash_11.jpg'
 
#roi = cv2.imread('images/soccer-player-2.jpg')
#roi = cv2.imread('images/object_group_2.jpg')
roi = cv2.imread(path)

hsv = cv2.cvtColor(roi,cv2.COLOR_BGR2HSV)
 
#target = cv2.imread('images/soccer-player-2.jpg')
#target = cv2.imread('images/surgeon_2.jpg')
#target = cv2.imread('images/object_group_2.jpg')
target = cv2.imread(path)
target = imutilspy.resize(target, height = 1000)

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
cnt_full = target.copy()

# Code to draw the contours
__ , contours, hierarchy = cv2.findContours(thresh_one.copy(),cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
cnts = sorted(contours, key = cv2.contourArea, reverse = True)



print time() - time_1

size_min = 200
size_max = 8000
visual = False

cnt_target = view_all_contours(target, size_min, size_max, visual)
cv2.drawContours(cnt_full, cnts, -1,(0,0,255),2)


res = imutilspy.resize(thresh_one, height = 700)
cv2.imshow('Original image', target)
cv2.imshow('Preprocessed', thresh_one)
cv2.imshow('All contours', cnt_full)
cv2.imshow('Filtered contours', cnt_target)

cv2.waitKey(0)



