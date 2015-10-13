#!/usr/local/bin/python2.7
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

from sklearn import cross_validation
from sklearn.metrics import accuracy_score,accuracy_score, confusion_matrix


# Initate ORB detector object
orb = cv2.ORB()

# Get the path of the training set
#parser = ap.ArgumentParser()
#parser.add_argument("-t", "--trainingSet", help="Path to Training Set", required="True")
#args = vars(parser.parse_args())

# Get the training classes names and store them in a list
train_path = 'dataset/data/'
training_names = os.listdir(train_path)

# Get all the path to the images and save them in a list
# image_paths and the corresponding label in image_paths
image_paths_tr = []
image_classes_tr = []
class_id = 0

#####
#features_train, features_test, labels_train, labels_test = cross_validation.train_test_split(main, target, test_size = 0.2)
####
print 'Saving image paths as a list'
for training_name in training_names:
    dir = os.path.join(train_path, training_name)
    class_path = imutils.imlist(dir)
    image_paths_tr+=class_path
    image_classes_tr+=[class_id]*len(class_path)
    class_id+=1
    
image_paths_tr, image_paths_te, image_classes_tr, image_classes_te = cross_validation.train_test_split(image_paths_tr, image_classes_tr, test_size = 0.2)



# List where all the descriptors are stored
des_list = []
print 'Iterating through features'
for image_path in image_paths_tr:
    #print image_path
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
im_features = np.zeros((len(image_paths_tr), k), "float32")
for i in xrange(len(image_paths_tr)):
    words, distance = vq(des_list[i][1],voc)
    for w in words:
        im_features[i][w] += 1

# Perform Tf-Idf vectorization
nbr_occurences = np.sum( (im_features > 0) * 1, axis = 0)
idf = np.array(np.log((1.0*len(image_paths_tr)+1) / (1.0*nbr_occurences + 1)), 'float32')

# Scaling the words
print 'Scaling Words. . .'
stdSlr = StandardScaler().fit(im_features)
im_features = stdSlr.transform(im_features)

# Train the Linear SVM
print 'Training SVM. . .'
clf = LinearSVC()
clf.fit(im_features, np.array(image_classes_tr))

# Save the SVM
print 'Saving SVM model. . .'
joblib.dump((clf, training_names, stdSlr, k, voc), "bof_cuav.pkl", compress=3)    
print 'Done!'    




############# Entering predicting phase ###############





# Load the classifier, class names, scaler, number of clusters and vocabulary 
clf, classes_names, stdSlr, k, voc = joblib.load("bof_uav.pkl")

# Get the path of the testing set
'''
parser = ap.ArgumentParser()
group = parser.add_mutually_exclusive_group(required=True)
group.add_argument("-t", "--testingSet", help="Path to testing Set")
group.add_argument("-i", "--image", help="Path to image")
parser.add_argument('-v',"--visualize", action='store_true')
args = vars(parser.parse_args())
'''

# Ask for user input

response = 't'
'''
print 'Press to test the entire set [t] or individual images [i]:'
response = raw_input()
'''



test_paths = 'dataset/our_test'
visualize = True

# Get the path of the testing image(s) and store them in a list
image_paths = []
if response == 't':
    test_path = 'dataset/our_test'
    try:
        testing_names = os.listdir(test_path)
    except OSError:
        print "No such directory {}\nCheck if the file exists".format(test_path)
        exit()
    for testing_name in testing_names:
        dir = os.path.join(test_path, testing_name)
        class_path = imutils.imlist(dir)
        image_paths+=class_path
else:
    image_paths = ['dataset/test']
    
# Create feature extraction and keypoint detector objects
'''
fea_det = cv2.FeatureDetector_create("SIFT")
des_ext = cv2.DescriptorExtractor_create("SIFT")
'''

# Initate ORB detector object
orb = cv2.ORB()


# List where all the descriptors are stored
des_list = []

for image_path in image_paths_te:
    im = cv2.imread(image_path)
    if im == None:
        print "No such file {}\nCheck if the file exists".format(image_path)
        exit()
    #kpts = fea_det.detect(im)
    kpts = orb.detect(im,None)
    
    #kpts, des = des_ext.compute(im, kpts)
    kpts, des = orb.compute(im, kpts)
    
    des_list.append((image_path, des))   
    
# Stack all the descriptors vertically in a numpy array
descriptors = des_list[0][1]
for image_path, descriptor in des_list[0:]:
    descriptors = np.vstack((descriptors, descriptor)) 

# 
test_features = np.zeros((len(image_paths_te), k), "float32")
for i in xrange(len(image_paths_te)):
    words, distance = vq(des_list[i][1],voc)
    for w in words:
        test_features[i][w] += 1

# Perform Tf-Idf vectorization
nbr_occurences = np.sum( (test_features > 0) * 1, axis = 0)
idf = np.array(np.log((1.0*len(image_paths)+1) / (1.0*nbr_occurences + 1)), 'float32')

# Scale the features
test_features = stdSlr.transform(test_features)

# Perform the predictions
predictions =  [classes_names[i] for i in clf.predict(test_features)]

# Visualize the results, if "visualize" flag set to true by the user
if visualize:
    for image_path, prediction in zip(image_paths_te, predictions):
        image = cv2.imread(image_path)
        cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
        pt = (0, 3 * image.shape[0] // 4)
        cv2.putText(image, prediction, pt ,cv2.FONT_HERSHEY_SCRIPT_COMPLEX, 2, [0, 255, 0], 2)
        cv2.imshow("Image", image)
        print '- ', prediction
        cv2.waitKey(3000)

