import numpy as np
from scipy.sparse import csr_matrix
import gzip
import csv
import lpputils
import datetime
import matplotlib.pyplot as plt
import sklearn.linear_model as skllm
import pandas as pd
from predtekmovanje import ModelPred


class Model(ModelPred):
	def train(self, x, y, routes, alpha = 1):
		self.routes = routes.unique()
		self.models = {}
		for route in self.routes:
			current = routes == route
			self.models[route] = reg = skllm.Ridge(alpha, fit_intercept=True).fit(x[current], y[current])
		self.avg_model = skllm.Ridge(alpha, fit_intercept=True).fit(x, y)
	
	def predict_duration(self, x, routes):
		y = np.zeros(len(x))
		for route in self.routes:
			current = routes == route
			# če v danih podatkih ni dane linije
			if current.sum() == 0: continue
			y[current] = self.models[route].predict(x[current])
		
		# za linije, ki niso bile v učnih podatkih, uporabi povprečen model
		current = y == 0
		if current.sum() != 0:
			y[current] = self.avg_model.predict(x[current])

		return y
	
	def predict_arrival(self, x, start_t, routes):
		#prišteje trajanje k začetku
		return (start_t + pd.to_timedelta(self.predict_duration(x, routes), unit='seconds'))


if __name__ == '__main__':
	filename = "./train.csv"
	x, y, routes = ModelPred.open_train(filename)

	'''
	# delitev na učne in testne podatke za preverjanje
	test_size = 100000
	ch = np.random.choice(range(len(y)), size=(test_size,), replace=False)    
	ind_test = np.zeros((len(y)), dtype=bool)
	ind_test[ch] = True
	ind_train = ~ind_test

	model = Model()
	model.train(x[ind_train], y[ind_train], routes[ind_train], 1)
	xt, start_t, routes = Model.open_test(filename)
	yh = model.predict_duration(xt[ind_test], routes[ind_test])
	print(np.sum(np.abs(yh - y[ind_test]))/test_size)'''
	
	model = Model()
	model.train(x, y, routes, 1)
	
	filename_test = "./test.csv"
	xt, start_t, routes = Model.open_test(filename_test)

	predictions = model.predict_arrival(xt, start_t, routes)
	predictions.to_csv('tekmovanje.txt', header=False, sep=',', index=False, mode='wt', date_format=lpputils.FORMAT, quoting=csv.QUOTE_NONE)