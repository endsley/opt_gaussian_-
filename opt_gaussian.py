#!/usr/bin/env python


import sys
import matplotlib
import numpy as np
import random
import itertools
import socket
import sklearn.metrics
from scipy.optimize import minimize
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder
from debug import *

np.set_printoptions(precision=4)
np.set_printoptions(threshold=sys.maxsize)
np.set_printoptions(linewidth=300)
np.set_printoptions(suppress=True)



class opt_gaussian():
	def __init__(self, X, Y, data_name=''):	#	X=data, Y=label, q=reduced dimension
		ń = X.shape[0]
		ð = X.shape[1]
		self.Ⅱᵀ = np.ones((ń,ń))
		ńᒾ = ń*ń
		self.Ⲏ = np.eye(ń) - (1.0/ń)*self.Ⅱᵀ

		Yₒ = OneHotEncoder(categories='auto', sparse=False).fit_transform(np.reshape(Y,(len(Y),1)))
		self.Kᵧ = Yₒ.dot(Yₒ.T)
		ṉ = np.sum(self.Kᵧ)

		self.ɡ = ɡ = 1.0/ṉ
		self.ḡ = ḡ = 1.0/(ńᒾ - ṉ)

		self.Q = ḡ*self.Ⅱᵀ - (ɡ + ḡ)*self.Kᵧ

		Ð = sklearn.metrics.pairwise.pairwise_distances(X)
		self.σₒ = np.median(Ð)
		self.Ðᒾ = (-Ð*Ð)/2

		#self.result = minimize(self.maxKseparation, self.σₒ, method='BFGS', options={'gtol': 1e-6, 'disp': True})
		self.result = minimize(self.ℍ, self.σₒ, method='BFGS', options={'gtol': 1e-8, 'disp': True})

	def maxKseparation(self, σ):
		Kₓ = np.exp(self.Ðᒾ/(σ*σ))
		loss = np.sum(Kₓ*self.Q)
		return loss

	def ℍ(self, σ):
		Ⲏ = self.Ⲏ
		Kₓ = np.exp(self.Ðᒾ/(σ*σ))
		Kᵧ = self.Kᵧ

		loss = -np.sum((Kₓ.dot(Ⲏ))*(Kᵧ.dot(Ⲏ)))
		return loss


if __name__ == "__main__":
	data_name = 'wine_2'
	X = np.loadtxt('data/' + data_name + '.csv', delimiter=',', dtype=np.float64)			
	Y = np.loadtxt('data/' + data_name + '_label.csv', delimiter=',', dtype=np.int32)			
	X_test = np.loadtxt('data/' + data_name + '_test.csv', delimiter=',', dtype=np.float64)			
	Y_test = np.loadtxt('data/' + data_name + '_label_test.csv', delimiter=',', dtype=np.int32)			

	X = preprocessing.scale(X)
	X_test = preprocessing.scale(X_test)

	opt_σ = opt_gaussian(X,Y, data_name)	#q if not set, it is automatically set to 80% of data variance by PCA
	#maxKseparation_debug(opt_σ)
	ℍ_debug(opt_σ)

	#print(opt_σ.maxKseparation(3.028))
	#print(opt_σ.maxKseparation(3.039))

	#print('\n\n')

	#print(opt_σ.ℍ(3.028))
	#print(opt_σ.ℍ(3.039))


	#print(opt_σ.result.x)

