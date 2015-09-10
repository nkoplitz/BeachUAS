from PIL import Image
from pyimagesearch import imutils
import cv2
from time import time

img = Image.open('images/beach_trash_3.jpg')
img = img.convert("LA")
datas = img.getdata()

newData = []


start_time = time()
for item in datas:
    #if item[0] == 255 and item[1] == 255 and item[2] == 255:
    #if item[0] > 75 and item[0] < 225:  # Colors in background
    if item[0] < 75 or item[0] > 225:  # Colors in foreground objects
        #newData.append((255, 255, 255, 0))
        #newData.append((0, 0, 0, 0)) # Make the colors black
        newData.append((0,0)) # Make the colors black
    else:
        newData.append(item)
print("--- %s seconds ---" % (time() - start_time))

img.putdata(newData)
img.save("img2.png", "PNG")

img = cv2.imread('img2.png')
img = imutils.resize(img, height = 700)
cv2.imshow('result', img)
cv2.waitKey(0)

#----------------------------------------------------------#

img = Image.open('img2.png')
img = img.convert("RGBA")
datas = img.getdata()

newData = []
for item in datas:
    if item[0] == 0 and item[1] == 0 and item[2] == 0:
        newData.append((0, 0, 255, 0))
    else:
        newData.append(item)

img.putdata(newData)
img.save("img3.png", "PNG")
img = cv2.imread('img3.png')
img = imutils.resize(img, height = 700)
cv2.imshow('result', img)
cv2.waitKey(0)
#---------------------------------------------------------#
imgray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

# Contour detection
#ret,thresh = cv2.threshold(imgray,127,255,0)

#imgray = cv2.GaussianBlur(imgray, (5, 5), 200)
imgray = cv2.medianBlur(imgray, 11)

#cv2.imshow('Blurred', imgray)

thresh = imgray

#imgray = cv2.medianBlur(imgray,5)
#imgray = cv2.Canny(imgray,10,500)
thresh = cv2.Canny(imgray,75,200)
#thresh = imgray
cv2.imshow('Canny', thresh)

contours, hierarchy = cv2.findContours(thresh.copy(),cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

cnts = sorted(contours, key = cv2.contourArea, reverse = True)[:5]

im = cv2.imread('images/beach_trash_3.jpg')
im = imutils.resize(im, height = 700)
test = im.copy()
cv2.drawContours(test, cnts, -1,(0,255,0),2)
cv2.imshow('All contours', test)
cv2.waitKey(0)
