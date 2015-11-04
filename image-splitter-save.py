import cv2
import numpy as np
from pyimagesearch import imutilspy
import imutils
import os



# Get the training classes names and store them in a list
train_path = 'dataset/data/'
training_names = os.listdir(train_path)

# Get all the path to the images and save them in a list
# image_paths and the corresponding label in image_paths
image_paths_m = []
image_classes_m = []
class_id = 0

print 'Saving image paths as a list'
for training_name in training_names:
    dir = os.path.join(train_path, training_name)
    class_path = imutils.imlist(dir)
    image_paths_m+=class_path
    class_id+=1

    
# List where all the descriptors are stored


print 'Iterating through features'


m = 0
dest_path = 'dataset/4x4_data/water-bottle/'
for image_path in image_paths_m:
    print image_path
    im = cv2.imread(image_path)
    im = imutilspy.resize(im, height = 200)

    img_height = im.shape[0]
    img_width = im.shape[1]
    
    uple = im[0:(img_height)/2 , 0:(img_width)/2]
    dole = im[(img_height)/2:img_height , 0:(img_width)/2]
    upri = im[0:(img_height)/2 , ((img_width)/2):img_width]
    dori = im[((img_height)/2):img_height , ((img_width)/2):img_width]

    uple_st = dest_path +  str(m) + '_0.png'
    dole_st = dest_path + str(m) + '_1.png'
    upri_st = dest_path + str(m) +  '_2.png'
    dori_st = dest_path + str(m) + '_3.png'
    
    
    cv2.imwrite(uple_st,uple)
    cv2.imwrite(dole_st,dole)
    cv2.imwrite(upri_st,upri)
    cv2.imwrite(dori_st,dori)
    
    m = m + 1
print 'Done!'
