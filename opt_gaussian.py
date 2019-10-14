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
from sklearn.utils.random import sample_without_replacement
from debug import *
import time 

np.set_printoptions(precision=4)
np.set_printoptions(threshold=sys.maxsize)
np.set_printoptions(linewidth=300)
np.set_printoptions(suppress=True)



class opt_gaussian():
	def __init__(self, X, Y, σ_type='ℍ'):	#	X=data, Y=label, σ_type='ℍ', or 'maxKseparation'
		ń = X.shape[0]
		ð = X.shape[1]

		if ń > 200:
			#	Down sample first
			samples = sample_without_replacement(n_population=ń, n_samples=200)
			X = X[samples,:]
			Y = Y[samples]
			ń = X.shape[0]


		self.Ⅱᵀ = np.ones((ń,ń))
		ńᒾ = ń*ń
		Yₒ = OneHotEncoder(categories='auto', sparse=False).fit_transform(np.reshape(Y,(len(Y),1)))
		self.Kᵧ = Kᵧ = Yₒ.dot(Yₒ.T)
		ṉ = np.sum(Kᵧ)
		self.σ_type = σ_type

		Ð = sklearn.metrics.pairwise.pairwise_distances(X)
		self.σₒ = np.median(Ð)
		self.Ðᒾ = (-Ð*Ð)/2

		if σ_type == 'ℍ':
			Ⲏ = np.eye(ń) - (1.0/ń)*self.Ⅱᵀ
			self.Γ = Ⲏ.dot(Kᵧ).dot(Ⲏ)
			self.result = minimize(self.ℍ, self.σₒ, method='BFGS', options={'gtol': 1e-8, 'disp': True})
		else:
			self.ɡ = ɡ = 1.0/ṉ
			self.ḡ = ḡ = 1.0/(ńᒾ - ṉ)
	
			self.Q = ḡ*self.Ⅱᵀ - (ɡ + ḡ)*Kᵧ
			self.result = minimize(self.maxKseparation, self.σₒ, method='BFGS', options={'gtol': 1e-6, 'disp': True})


	def maxKseparation(self, σ):
		Kₓ = np.exp(self.Ðᒾ/(σ*σ))
		loss = np.sum(Kₓ*self.Q)
		return loss

	def ℍ(self, σ):
		Kₓ = np.exp(self.Ðᒾ/(σ*σ))
		Kᵧ = self.Kᵧ
		Γ = self.Γ

		loss = -np.sum(Kₓ*Γ)
		return loss




	def debug(self):
		if self.σ_type == 'ℍ':
			ℍ_debug(self)
		else:
			maxKseparation_debug(self)


def get_opt_σ(X,Y):
	optimizer = opt_gaussian(X,Y)	
	return optimizer.result.x



if __name__ == "__main__":
	data_name = 'car'
	X = np.loadtxt('data/' + data_name + '.csv', delimiter=',', dtype=np.float64)			
	Y = np.loadtxt('data/' + data_name + '_label.csv', delimiter=',', dtype=np.int32)			
	X_test = np.loadtxt('data/' + data_name + '_test.csv', delimiter=',', dtype=np.float64)			
	Y_test = np.loadtxt('data/' + data_name + '_label_test.csv', delimiter=',', dtype=np.int32)			

	X = preprocessing.scale(X)
	X_test = preprocessing.scale(X_test)

	start_time = time.time() 
	opt_σ = opt_gaussian(X,Y, σ_type='ℍ')	#σ_type='ℍ' or 'maxKseparation'
	print("--- %s seconds ---" % (time.time() - start_time))

	opt_σ.debug()

