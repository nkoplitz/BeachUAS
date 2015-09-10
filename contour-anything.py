from pyimagesearch.transform import four_point_transform
from matplotlib import pyplot as plt
import numpy as np
import cv2
import imutils

#im = cv2.imread('images/star-simple.jpg')
#im = cv2.imread('images/lego.png')
#im = cv2.imread('images/rectangle-skewed.jpg')
#im = cv2.imread('images/rectangle-jagged.jpg')
#im = cv2.imread('images/multi-rectangles.jpg')
#im = cv2.imread('images/square_1.jpg')
#im = cv2.imread('images/cube_0.jpg')
#im = cv2.imread('images/cube_1.png')
#im = cv2.imread('images/ing.png')
#im = cv2.imread('images/page.jpg')
#im = cv2.imread('images/multi-objects.png')
#im = cv2.imread('images/crowd-four.jpg')
#im = cv2.imread('images/car_three.jpg')
#im = cv2.imread('images/anthony-1.jpg')
#im = cv2.imread('images/car_two.jpg')
im = cv2.imread('images/beach_trash_3.jpg')
#im = cv2.imread('images/circles1.png')
#im = cv2.imread('images/waterbottle_0.jpg')

#cv2.imshow('Original', im)

# Histogram equalization to improve contrast




###
#im = np.fliplr(im)

im = imutils.resize(im, height = 500)

imgray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)

# Contour detection
#ret,thresh = cv2.threshold(imgray,127,255,0)

#imgray = cv2.GaussianBlur(imgray, (5, 5), 200)
imgray = cv2.medianBlur(imgray, 11)

cv2.imshow('Blurred', imgray)

'''
hist,bins = np.histogram(imgray.flatten(),256,[0,256])
plt_one = plt.figure(1)
cdf = hist.cumsum()
cdf_normalized = cdf * hist.max()/ cdf.max()

cdf_m = np.ma.masked_equal(cdf,0)
cdf_m = (cdf_m - cdf_m.min())*255/(cdf_m.max()-cdf_m.min())
cdf = np.ma.filled(cdf_m,0).astype('uint8')
imgray = cdf[imgray]

cv2.imshow('Histogram Normalization', imgray)
'''
'''
imgray = cv2.adaptiveThreshold(imgray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
            cv2.THRESH_BINARY,11,2)
'''

thresh = imgray

#imgray = cv2.medianBlur(imgray,5)
#imgray = cv2.Canny(imgray,10,500)
thresh = cv2.Canny(imgray,75,200)
#thresh = imgray
cv2.imshow('Canny', thresh)

'''
### Adding our circle code ###

cimg = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)

circles = cv2.HoughCircles(thresh, cv2.cv.CV_HOUGH_GRADIENT, 1.2, 100)

#circles = np.uint16(np.around(circles))
if circles is not None:
    for i in circles[0,:]:
        # draw the outer circle
        cv2.circle(cimg,(i[0],i[1]),i[2],(0,255,0),2)
        # draw the center of the circle
        cv2.circle(cimg,(i[0],i[1]),2,(0,0,255),3)

    cv2.imshow('detected circles',cimg)
else:
    print 'No circles detected'
##############################
'''


contours, hierarchy = cv2.findContours(thresh.copy(),cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

cnts = sorted(contours, key = cv2.contourArea, reverse = True)[:5]

test = im.copy()
cv2.drawContours(test, cnts, -1,(0,255,0),2)
cv2.imshow('All contours', test)

print '---------------------------------------------'
#####  Code to show each contour #####
main = np.array([[]])
for c in cnts:
    epsilon = 0.02*cv2.arcLength(c,True)
    approx = cv2.approxPolyDP(c,epsilon,True)

    test = im.copy()
    cv2.drawContours(test, [approx], -1,(0,255,0),2)
    #print 'Contours: ', contours
    if len(approx) == 4:
        print 'Found rectangle'
        print 'Approx.shape: ', approx.shape
        print 'Test.shape: ', test.shape

        # frame_f = frame_f[y: y+h, x: x+w]
        frame_f = test[approx[0,0,1]:approx[2,0,1], approx[0,0,0]:approx[2,0,0]]

        print 'frame_f.shape: ', frame_f.shape
        main = np.append(main, approx[None,:][None,:])
        print 'main: ', main


    # Uncomment in order to show all rectangles in image
    #cv2.imshow('Show Ya', test)

        
    #print 'Approx: ', approx.shape
    #cv2.imshow('Show Ya', frame_f)
    cv2.waitKey()
print '---------------------------------------------'
cv2.drawContours(im, cnts, -1,(0,255,0),2)
print main.shape
print main
cv2.imshow('contour-test', im)
cv2.waitKey()
########################################


'''
########## Actual scan.py code ###########

for c in cnts:
    epsilon = 0.02*cv2.arcLength(c,True)
    approx = cv2.approxPolyDP(c,epsilon,True)
    
    if len(approx) == 4:
        rectCnt = approx
        print rectCnt
        break
#cv2.drawContours(im, contours, -1,(0,255,0),2)
cv2.drawContours(im, [rectCnt], -1,(0,255,0),2)

cv2.imshow('contour-test', im)
cv2.waitKey()
##########################################
'''




'''
############## Adding Warp Transform code ###################

# apply the four point transform to obtain a top-down
# view of the original image
warped = four_point_transform(im, cnt.reshape(4, 2) * ratio)

# convert the warped image to grayscale, then threshold it
# to give it that 'black and white' paper effect
warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
warped = threshold_adaptive(warped, 250, offset = 10)
warped = warped.astype("uint8") * 255

cv2.imshow('Warped', im)
cv2.waitKey()
#############################################################
'''

