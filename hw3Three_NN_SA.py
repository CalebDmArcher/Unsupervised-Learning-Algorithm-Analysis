from numpy import genfromtxt
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.utils import shuffle
import random
import time

#read csv file and store data into numpy array
my_data = genfromtxt('sportsArticle02.csv', delimiter=',', skip_header=1)
# random.shuffle(my_data)
my_data = shuffle(my_data)

data_train = my_data[0:599]
data_verification = my_data[600:799]
data_testing = my_data[800:999]
#print(my_data.shape)
y_train = data_train[:,0]
y_verification = data_verification[:,0]
y_testing = data_testing[:,0]

X = my_data[:,1:60]
X_train = data_train[:,1:60]
X_verification = data_verification[:,1:60]
X_testing = data_testing[:,1:60]

# print(X.shape)
# print(y_train)
# print(y_verification)
# print(y_testing)


#do machine learning
#clf = KMeans(n_clusters=8,init='k-means++', n_init=10).fit(X_train , y_train)
attrN = 9
cluNKmean = 6
cluNGMM = 3

layerN = 200
unitN = 500

print('==========START=========')
print(' ')

t0 = time.time()
clf = MLPClassifier(hidden_layer_sizes=(layerN, unitN)).fit(X_train , y_train)
t1 = time.time()
print('Time for NN is {:.2f}'.format(t1 - t0))
print('Accuracy of Decision Tree classifier on training set: {:.2f}'
    .format(clf.score(X_train, y_train)))
print('Accuracy of Decision Tree classifier on verification set: {:.2f}'
    .format(clf.score(X_verification, y_verification)))
print('Accuracy of Decision Tree classifier on testing set: {:.2f}'
    .format(clf.score(X_testing, y_testing)))
# print(clf2.predict(X_testing))

t2 = time.time()
pcaX = PCA(n_components=attrN).fit_transform(X)
t2_2 = time.time()
pcaX_train = pcaX[0:599]
pcaX_verification = pcaX[600:799]
pcaX_testing = pcaX[800:999]
t2_3 = time.time()

print(" ")

clf2 = MLPClassifier(hidden_layer_sizes=(layerN,unitN)).fit(pcaX_train , y_train)
t3 = time.time()
print('Time for NN via PCA is {:.2f}'.format(t3 - t2_3 + t2_2 - t2))
print('Accuracy of Decision Tree classifier on training set: {:.2f}'
    .format(clf2.score(pcaX_train, y_train)))
print('Accuracy of Decision Tree classifier on verification set: {:.2f}'
    .format(clf2.score(pcaX_verification, y_verification)))
print('Accuracy of Decision Tree classifier on testing set: {:.2f}'
    .format(clf2.score(pcaX_testing, y_testing)))
# print(clf.predict(pcaX_testing))

print(" ")

t4 = time.time()
kMeanX = KMeans(n_clusters=cluNKmean).fit_transform(X)
t4_2 = time.time()
kMeanX_train = kMeanX[0:599]
kMeanX_verification = kMeanX[600:799]
kMeanX_testing = kMeanX[800:999]
t4_3 = time.time()
clf3 = MLPClassifier(hidden_layer_sizes=(layerN, unitN)).fit(kMeanX_train , y_train)
t5 = time.time()
print('Time for NN via K-mean is {:.2f}'.format(t5 - t4_3 + t4_2 - t4))
print('Accuracy of Decision Tree classifier on training set: {:.2f}'
    .format(clf3.score(kMeanX_train, y_train)))
print('Accuracy of Decision Tree classifier on verification set: {:.2f}'
    .format(clf3.score(kMeanX_verification, y_verification)))
print('Accuracy of Decision Tree classifier on testing set: {:.2f}'
    .format(clf3.score(kMeanX_testing, y_testing)))

print(' ')

t6 = time.time()
GMMX = GaussianMixture(n_components=cluNGMM).fit(X).predict_proba(X)
t6_2 = time.time()
GMMX_train = GMMX[0:599]
GMMX_verification = GMMX[600:799]
GMMX_testing = GMMX[800:999]
t6_3 = time.time()
clf = MLPClassifier(hidden_layer_sizes=(layerN, unitN)).fit(GMMX_train , y_train)
t7 = time.time()
print('Time for NN via GMM is {:.2f}'.format(t7 - t6_3 + t6_2 - t6))
print('Accuracy of Decision Tree classifier on training set: {:.2f}'
    .format(clf.score(GMMX_train, y_train)))
print('Accuracy of Decision Tree classifier on verification set: {:.2f}'
    .format(clf.score(GMMX_verification, y_verification)))
print('Accuracy of Decision Tree classifier on testing set: {:.2f}'
    .format(clf.score(GMMX_testing, y_testing)))


