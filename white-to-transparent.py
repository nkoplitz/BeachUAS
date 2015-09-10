from PIL import Image
from pyimagesearch import imutils
import cv2
from time import time

img = Image.open('images/beach_trash_3.jpg')

img = img.convert("RGBA")
datas = img.getdata()

newData = []


start_time = time()
for item in datas:
    #if item[0] == 255 and item[1] == 255 and item[2] == 255:
    if item[0] > 200 and item[1] > 200 and item[2] > 200:
        #newData.append((255, 255, 255, 0))
        #newData.append((0, 0, 0, 0)) # Make the colors black
        newData.append((255, 0, 0, 0)) # Make the colors red
    else:
        newData.append(item)
print("--- %s seconds ---" % (time() - start_time))

img.putdata(newData)
img.save("img2.png", "PNG")

img = cv2.imread('img2.png')
img = imutils.resize(img, height = 700)
cv2.imshow('result', img)
cv2.waitKey(0)
