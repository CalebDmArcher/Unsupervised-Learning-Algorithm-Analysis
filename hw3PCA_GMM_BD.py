from numpy import genfromtxt
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
import random

#read csv file and store data into numpy array
my_data = genfromtxt('bloodDonation02.csv', delimiter=',', skip_header=1)
random.shuffle(my_data)

data_train = my_data[0:479]
data_verification = my_data[480:599]
data_testing = my_data[600:748]

y_train = data_train[:,0]
y_verification = data_verification[:,0]
y_testing = data_testing[:,0]

X = my_data[:,1:5]
X_train = data_train[:,1:5]
X_verification = data_verification[:,1:5]
X_testing = data_testing[:,1:5]

attrN = 3
pcaX = PCA(n_components=attrN).fit_transform(X)
gmm = GaussianMixture(n_components=20).fit(pcaX)
print(gmm.score(pcaX))
print(gmm.bic(pcaX))
