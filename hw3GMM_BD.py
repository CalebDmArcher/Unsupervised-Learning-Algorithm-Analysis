from numpy import genfromtxt
import numpy as np
from sklearn.mixture import GaussianMixture
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

#print(X)

gmm = GaussianMixture(n_components=20).fit(X)
print(gmm.score(X))
print(gmm.bic(X))
