#!/usr/local/bin/python2.7
import numpy as np
import cv2
from matplotlib import pyplot as plt
import imutils
from pyimagesearch import imutilspy
from time import time
import os

from sklearn.svm import LinearSVC
from sklearn.externals import joblib
from scipy.cluster.vq import *
from sklearn.preprocessing import StandardScaler

from sklearn import cross_validation
from sklearn.metrics import accuracy_score,accuracy_score, confusion_matrix

from sklearn.linear_model import Ridge
from sklearn.learning_curve import validation_curve, learning_curve
from sklearn.svm import SVC
from sklearn import svm

from sklearn.cross_validation import ShuffleSplit
from sklearn.grid_search import GridSearchCV


def main_mat_construct(image_paths_m):
    print 'Iterating through features. . .'
    for image_path in image_paths_m:
        #print image_path
        im = cv2.imread(image_path)
        im = imutilspy.resize(im, height = 200)
        
        #kpts = fea_det.detect(im)
        kpts = orb.detect(im,None)
        
        #kpts, des = des_ext.compute(im, kpts)
        kp, des = orb.compute(im, kpts)
        
        des_list.append((image_path, des))
        
        
    # Stack all the descriptors vertically in a numpy array
    print 'Converting data to main matrix. . .'
    descriptors = des_list[0][1]
    for image_path, descriptor in des_list[1:]:
        descriptors = np.vstack((descriptors, descriptor))  
    # Perform k-means clustering
    k = 100
    voc, variance = kmeans(descriptors, k, 1) 

    # Calculate the histogram of features
    print 'Calculating histogram of features. . .'
    im_features = np.zeros((len(image_paths_m), k), "float32")
    for i in xrange(len(image_paths_m)):
        words, distance = vq(des_list[i][1],voc)
        for w in words:
            im_features[i][w] += 1

    # Perform Tf-Idf vectorization
    nbr_occurences = np.sum( (im_features > 0) * 1, axis = 0)
    idf = np.array(np.log((1.0*len(image_paths_m)+1) / (1.0*nbr_occurences + 1)), 'float32')

    # Scaling the words
    print 'Scaling Words. . .'
    stdSlr = StandardScaler().fit(im_features)
    im_features = stdSlr.transform(im_features)

    return im_features

def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt

def grabCut(img):
    
    mask = np.zeros(img.shape[:2],np.uint8)
    bgdModel = np.zeros((1,65),np.float64)
    fgdModel = np.zeros((1,65),np.float64)

    #rect = (50,50,450,290)
    rect = (0,0,img.shape[0],img.shape[1])
    cv2.grabCut(img,mask,rect,bgdModel,fgdModel,5,cv2.GC_INIT_WITH_RECT)
    mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
    img = img*mask2[:,:,np.newaxis]
    return img

t0 = time()

# Initate ORB detector object
orb = cv2.ORB_create()

# Get the path of the training set
#parser = ap.ArgumentParser()
#parser.add_argument("-t", "--trainingSet", help="Path to Training Set", required="True")
#args = vars(parser.parse_args())

# Get the training classes names and store them in a list

train_path = 'dataset/data_two/'
#train_path = 'dataset/data_two_choice/train/'
#train_path = 'caltech_dataset/data/'

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
    image_classes_m+=[class_id]*len(class_path)
    class_id+=1
    
image_paths_tr, image_paths_te, image_classes_tr, image_classes_te = cross_validation.train_test_split(image_paths_m, image_classes_m, test_size = 0.2)


# List where all the descriptors are stored
des_list = []
sk_count = 0
count = 0
print 'Iterating through features'
for image_path in image_paths_tr:
    im = cv2.imread(image_path)
    im = imutilspy.resize(im, height = 200)

    im = grabCut(im)
    
    #kpts = fea_det.detect(im)
    kpts = orb.detect(im,None)
    
    #kpts, des = des_ext.compute(im, kpts)
    kp, des = orb.compute(im, kpts)
    
    
    if des == None:
        print image_path
        os.remove(image_path)
        sk_count = sk_count + 1
    else:
        des_list.append((image_path, des))
    count = count + 1

#print sk_count, ' out of ', count, ' skipped'
    
# Stack all the descriptors vertically in a numpy array
print 'Converting data to matrix. . .'
descriptors = des_list[0][1]
for image_path, descriptor in des_list[1:]:

    if descriptor == None:
        print 'Skipped ', image_path, '. . .'
        continue
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
#clf = LinearSVC()
clf = svm.SVC(kernel='poly')
clf.fit(im_features, np.array(image_classes_tr))

# Save the SVM
print 'Saving SVM model. . .'
joblib.dump((clf, training_names, stdSlr, k, voc), "bof_cuav.pkl", compress=3)    
print 'Done!'    

total_time = time() - t0
print total_time, 's'






############# Entering predicting phase ###############


print 'Entering prediction phase. . .'
t0 = time()

# Load the classifier, class names, scaler, number of clusters and vocabulary 
clf, classes_names, stdSlr, k, voc = joblib.load("bof_cuav.pkl")

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
orb = cv2.ORB_create()


# List where all the descriptors are stored
des_list = []

for image_path in image_paths_te:
    im = cv2.imread(image_path)
    im = imutilspy.resize(im, height = 200)

    im = grabCut(im)
    
    if im == None:
        print "No such file {}\nCheck if the file exists".format(image_path)
        exit()
    #kpts = fea_det.detect(im)
    kpts = orb.detect(im,None)
    
    #kpts, des = des_ext.compute(im, kpts)
    kpts, des = orb.compute(im, kpts)
    cv2.imshow('test',im)
    cv2.waitKey(0)
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

class_num = [0,1,2]

# Perform the predictions
predictions =  [classes_names[i] for i in clf.predict(test_features)]

predictions_1 =  [class_num[i] for i in clf.predict(test_features)]

# Visualize the results, if "visualize" flag set to true by the user

accuracy = accuracy_score(image_classes_te, predictions_1)

print 'Finished'
total_time = time() - t0
print total_time, 's'
print ''
print ''

print 'Confusion Matrix: '
print confusion_matrix(image_classes_te, predictions_1)

# Accuracy in the 0.9333, 9.6667, 1.0 range
print 'Score: ',accuracy


while True:
    print 'Visualize the classifications? [y] or [n]'
    response = raw_input()
    if response == 'y' or response == 'n':
        if response == 'n':
            visualize = False
        break

if visualize:
    for image_path, prediction in zip(image_paths_te, predictions):
        image = cv2.imread(image_path)
        cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
        pt = (0, 3 * image.shape[0] // 4)
        cv2.putText(image, prediction, pt ,cv2.FONT_HERSHEY_SCRIPT_COMPLEX, 2, [0, 255, 0], 2)
        image = imutilspy.resize(image, height = 400)
        cv2.imshow("Image", image)
        print '- ', prediction
        cv2.waitKey(3000)





print ''
print ''
print 'Press [p] to plot validation and learning curve or else press anything else: '
plot_option = raw_input()

if plot_option == 'p' or plot_option == 't':
    t0 = time()
    print 'Plotting validation learning curve. . .'
    # SVM model
    estimator = clf

    
    cv = ShuffleSplit(im_features.shape[0], n_iter=10, test_size=0.2, random_state=0)
    gammas = np.logspace(-6, -1, 10)
    classifier = GridSearchCV(estimator=estimator, cv=cv, param_grid=dict(gamma=gammas))
    classifier.fit(im_features, image_classes_tr)
    title = 'Learning Curves (SVM, linear kernel, $\gamma=%.6f$)' %classifier.best_estimator_.gamma
    estimator = SVC(kernel='linear', gamma=classifier.best_estimator_.gamma)

    if plot_option == 'p':
        plot_learning_curve(estimator, title, im_features, np.array(image_classes_tr), cv=cv)
        print 'Finished'
        total_time = time() - t0
        print total_time, 's'
        plt.show()


    t0 = time()    
    #print 'Constructing main matrix. . .'
    #main_mat = main_mat_construct(image_paths_m)
    
    print 'Plotting validation curve. . .'
    param_range = np.logspace(-6, -1, 5)
    '''
    train_scores, test_scores = validation_curve(
        SVC(), main_mat, image_classes_m, param_name="gamma", param_range=param_range,
        cv=10, scoring="accuracy", n_jobs=1)
    '''
    train_scores, test_scores = validation_curve(
        SVC(), im_features, np.array(image_classes_tr), param_name="gamma", param_range=param_range,
        cv=10, scoring="accuracy", n_jobs=1)

    
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    print 'Variables assigned'
    print 'Preparing plot. . .'
    plt.title("Validation Curve with SVM")
    plt.xlabel("$\gamma$")
    plt.ylabel("Score")
    plt.ylim(0.0, 1.1)
    plt.semilogx(param_range, train_scores_mean, label="Training score", color="r")
    plt.fill_between(param_range, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.2, color="r")
    plt.semilogx(param_range, test_scores_mean, label="Cross-validation score",
                 color="g")
    plt.fill_between(param_range, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.2, color="g")
    plt.legend(loc="best")

    if plot_option == 't':
        plot_learning_curve(estimator, title, im_features, np.array(image_classes_tr), cv=cv)


    print 'Finished'
    total_time = time() - t0
    print total_time, 's'
    
    plt.show()










