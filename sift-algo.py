import cv2
image = cv2.imread('images/car_two.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
sift = cv2.xfeatures2d.SIFT_create()
