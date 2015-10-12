#!/usr/local/bin/python2.7

import argparse as ap
import numpy as np
import cv2
from matplotlib import pyplot as plt
import imutils 
import numpy as np
import os
from sklearn.svm import LinearSVC
from sklearn.externals import joblib
from scipy.cluster.vq import *
from sklearn.preprocessing import StandardScaler

# Initate ORB detector object
orb = cv2.ORB()

# Get the path of the training set
#parser = ap.ArgumentParser()
#parser.add_argument("-t", "--trainingSet", help="Path to Training Set", required="True")
#args = vars(parser.parse_args())

# Get the training classes names and store them in a list
train_path = 'dataset/our_train/'
training_names = os.listdir(train_path)

# Get all the path to the images and save them in a list
# image_paths and the corresponding label in image_paths
image_paths = []
image_classes = []
class_id = 0

print 'Saving image paths as a list'
for training_name in training_names:
    dir = os.path.join(train_path, training_name)
    class_path = imutils.imlist(dir)
    image_paths+=class_path
    image_classes+=[class_id]*len(class_path)
    class_id+=1

# Create feature extraction and keypoint detector objects
#print 'Creating SIFT classifiers'
#fea_det = cv2.FeatureDetector_create("SIFT")
#des_ext = cv2.DescriptorExtractor_create("SIFT")

# List where all the descriptors are stored
des_list = []
print 'Iterating through features'
for image_path in image_paths:
    print image_path
    im = cv2.imread(image_path)
    
    #kpts = fea_det.detect(im)
    kpts = orb.detect(im,None)
    
    #kpts, des = des_ext.compute(im, kpts)
    kp, des = orb.compute(im, kpts)
    
    des_list.append((image_path, des))
    
    
# Stack all the descriptors vertically in a numpy array
print 'Converting data to matrix. . .'
descriptors = des_list[0][1]
for image_path, descriptor in des_list[1:]:
    descriptors = np.vstack((descriptors, descriptor))  

# Perform k-means clustering
k = 100
voc, variance = kmeans(descriptors, k, 1) 

# Calculate the histogram of features
print 'Calculating histogram of features. . .'
im_features = np.zeros((len(image_paths), k), "float32")
for i in xrange(len(image_paths)):
    words, distance = vq(des_list[i][1],voc)
    for w in words:
        im_features[i][w] += 1

# Perform Tf-Idf vectorization
nbr_occurences = np.sum( (im_features > 0) * 1, axis = 0)
idf = np.array(np.log((1.0*len(image_paths)+1) / (1.0*nbr_occurences + 1)), 'float32')

# Scaling the words
print 'Scaling Words. . .'
stdSlr = StandardScaler().fit(im_features)
im_features = stdSlr.transform(im_features)

# Train the Linear SVM
print 'Training SVM. . .'
clf = LinearSVC()
clf.fit(im_features, np.array(image_classes))

# Save the SVM
print 'Saving SVM model. . .'
joblib.dump((clf, training_names, stdSlr, k, voc), "bof_uav.pkl", compress=3)    
print 'Done!'    
