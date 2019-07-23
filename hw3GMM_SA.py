from numpy import genfromtxt
#import numpy as np
from sklearn.mixture import GaussianMixture
import random

#read csv file and store data into numpy array
my_data = genfromtxt('sportsArticle02.csv', delimiter=',', skip_header=1)
#random.shuffle(my_data)

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

#do machine learning
#clf = KMeans(n_clusters=8,init='k-means++', n_init=10).fit(X_train , y_train)
gmm = GaussianMixture(n_components=6).fit(X)
print(gmm.bic(X))

