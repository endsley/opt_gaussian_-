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
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder

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

		Yₒ = OneHotEncoder(categories='auto', sparse=False).fit_transform(np.reshape(Y,(len(Y),1)))
		self.Kᵧ = Yₒ.dot(Yₒ.T)
		ṉ = np.sum(self.Kᵧ)

		self.ɡ = ɡ = 1.0/ṉ
		self.ḡ = ḡ = 1.0/(ńᒾ - ṉ)

		self.Q = ḡ*self.Ⅱᵀ - (ɡ + ḡ)*self.Kᵧ

		Ð = sklearn.metrics.pairwise.pairwise_distances(X)
		self.σₒ = np.median(Ð)
		self.Ðᒾ = (-Ð*Ð)/2

		self.result = minimize(self.foo, self.σₒ, method='BFGS', options={'gtol': 1e-6, 'disp': True})

	def foo(self, σ):
		Kₓ = np.exp(self.Ðᒾ/(σ*σ))
		loss = np.sum(Kₓ*self.Q)
		return loss


	def debug(self):
		Kᵧ = self.Kᵧ
		Q = self.Q
		ɡ = self.ɡ
		ḡ = self.ḡ
		Ⅱᵀ = self.Ⅱᵀ
		ƌₐ = []
		ƌᵦ = []
		lossⲷ = []
		σⲷ = np.arange(0.01,7, 0.01)

		for σ in σⲷ:
			Kₓ = np.exp(self.Ðᒾ/(σ*σ))
			ƌₐ.append(ɡ*np.sum(Kₓ*Kᵧ))
			ƌᵦ.append(ḡ*np.sum(Kₓ*(Ⅱᵀ - Kᵧ)))

			Δƌ = np.array(ƌₐ) - np.array(ƌᵦ)
			lossⲷ.append(self.foo(σ))
	
		loss = self.foo(self.result.x)
		lossₒ = self.foo(self.σₒ)

		print('σₒ = %.3f'%self.σₒ)
		print('σ = %.3f'%self.result.x)
		print('lossₒ = %.3f'%lossₒ)
		print('loss = %.3f'%loss)

		optσText = 'Optimal σ : %.3f\nopt kernel separation : %.3f'%(self.result.x, -loss)

		plt.plot(σⲷ, ƌₐ, 'r-')
		plt.plot(σⲷ, ƌᵦ, 'b-')
		plt.plot(σⲷ, Δƌ, 'g-')
		#plt.plot(σⲷ, lossⲷ, 'y-')
		plt.xlabel('σ value')
		plt.ylabel('Kernel Value')
		plt.title('Kernel Value as Varying σ')
		plt.text(σⲷ[-1], ƌₐ[-1], 'Mean within cluster kernel value', horizontalalignment='right')
		plt.text(σⲷ[-1], ƌᵦ[-1], 'Mean between cluster kernel value', horizontalalignment='right')
		plt.text(self.result.x, -loss, optσText, horizontalalignment='center')
		plt.axvline(x=self.result.x, linestyle="dashed")

		#plt.text(σⲷ[-1], Δƌ[-1], 'Kernel Value Δ', horizontalalignment='right')

		plt.show()

if __name__ == "__main__":
	data_name = 'wine_2'
	X = np.loadtxt('data/' + data_name + '.csv', delimiter=',', dtype=np.float64)			
	Y = np.loadtxt('data/' + data_name + '_label.csv', delimiter=',', dtype=np.int32)			
	X_test = np.loadtxt('data/' + data_name + '_test.csv', delimiter=',', dtype=np.float64)			
	Y_test = np.loadtxt('data/' + data_name + '_label_test.csv', delimiter=',', dtype=np.int32)			

	X = preprocessing.scale(X)
	X_test = preprocessing.scale(X_test)

	opt_σ = opt_gaussian(X,Y, data_name)	#q if not set, it is automatically set to 80% of data variance by PCA
	opt_σ.debug()

