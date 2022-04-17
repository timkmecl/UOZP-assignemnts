import numpy as np
from scipy.sparse import csr_matrix
import gzip
import csv
import lpputils
import datetime
import matplotlib.pyplot as plt
import sklearn.linear_model as skllm
import pandas as pd


class ModelPred():
	PRAZNIKI = pd.Series([ # samo tisti med tednom
		datetime.datetime(2012, 1, 2),
		datetime.datetime(2012, 2, 8),
		datetime.datetime(2012, 4, 8),
		datetime.datetime(2012, 4, 27),
		datetime.datetime(2012, 5, 1),
		datetime.datetime(2012, 5, 2),
		datetime.datetime(2012, 6, 25),
		datetime.datetime(2012, 8, 15),
		datetime.datetime(2012, 10, 31),
		datetime.datetime(2012, 11, 1),
		datetime.datetime(2012, 12, 25),
		datetime.datetime(2012, 12, 26),
	]).dt.dayofyear
	POCITNICE = pd.DataFrame([
		[datetime.datetime(2012, 2, 20), datetime.datetime(2012, 2, 24)],
		[datetime.datetime(2012, 4, 27), datetime.datetime(2012, 5, 2)],
		[datetime.datetime(2012, 6, 23), datetime.datetime(2012, 9, 3)],
		[datetime.datetime(2012, 6, 23), datetime.datetime(2012, 9, 3)],
		[datetime.datetime(2012, 12, 24), datetime.datetime(2013, 1, 1)],
	], columns=['start', 'end'])
	POCITNICE.start = POCITNICE.start.dt.dayofyear
	POCITNICE.end = POCITNICE.end.dt.dayofyear

	@staticmethod
	def open_train(filename):
		df = pd.read_csv(filename, sep="\t")
		df = df[['Route Direction', 'Departure time', 'Arrival time']]
		df = df.rename(columns = {'Route Direction':'route', 'Departure time':'start', 'Arrival time':'end'})
		
		df.start = pd.to_datetime(df.start, format=lpputils.FORMAT)
		df.end = pd.to_datetime(df.end, format=lpputils.FORMAT)

		y  = ((df.end - df.start)/pd.Timedelta(seconds=1)).astype(int)
		x = ModelPred.transform_data_for_linreg(df)
		route = df.route

		return x, y, route

	@staticmethod
	def open_test(filename):
		df = pd.read_csv(filename, sep="\t")
		df = df[['Route Direction', 'Departure time']]
		df = df.rename(columns = {'Route Direction':'route', 'Departure time':'start'})
		
		df.start = pd.to_datetime(df.start, format=lpputils.FORMAT)
		
		x = ModelPred.transform_data_for_linreg(df)

		route = df.route

		return x, df.start, route
	
	@staticmethod
	def transform_data_for_linreg(df):
		PRAZNIKI = pd.Series([ # samo tisti med tednom
			datetime.datetime(2012, 1, 2),
			datetime.datetime(2012, 2, 8),
			datetime.datetime(2012, 4, 8),
			datetime.datetime(2012, 4, 27),
			datetime.datetime(2012, 5, 1),
			datetime.datetime(2012, 5, 2),
			datetime.datetime(2012, 6, 25),
			datetime.datetime(2012, 8, 15),
			datetime.datetime(2012, 10, 31),
			datetime.datetime(2012, 11, 1),
			datetime.datetime(2012, 12, 25),
			datetime.datetime(2012, 12, 26),
		]).dt.dayofyear
		POCITNICE = pd.DataFrame([
			[datetime.datetime(2012, 2, 20), datetime.datetime(2012, 2, 24)],
			[datetime.datetime(2012, 4, 27), datetime.datetime(2012, 5, 2)],
			[datetime.datetime(2012, 6, 23), datetime.datetime(2012, 9, 3)],
			[datetime.datetime(2012, 6, 23), datetime.datetime(2012, 9, 3)],
			[datetime.datetime(2012, 12, 24), datetime.datetime(2013, 1, 1)],
		], columns=['start', 'end'])
		POCITNICE.start = POCITNICE.start.dt.dayofyear
		POCITNICE.end = POCITNICE.end.dt.dayofyear

		df['dayofyear'] = df.start.dt.dayofyear
		df['dan_v_tednu'] = df.start.dt.weekday
		# začetne čase razdeli v 5-minutne intervale
		df['cas'] = (df.start.dt.hour*60 + df.start.dt.minute) // 5

		df['pon_cet'] = df.dan_v_tednu < 4
		df['petek'] = df.dan_v_tednu == 4
		df['sobota'] = df.dan_v_tednu == 5
		df['nedelja'] = df.dan_v_tednu == 6
		df['praznik'] = df.dayofyear.isin(PRAZNIKI)

		# preveri, ali pade dan v letu znotraj intervala keterih od počitnic
		df['pocitnice'] = False
		for i, d in POCITNICE.iterrows():
			df.pocitnice = df.pocitnice | ((d.start <= df.dayofyear) & (d.end >= df.dayofyear))
		
		x = np.zeros((len(df), 6 + 24*12))
		x[:, 0:6] = df[['pon_cet', 'petek', 'sobota', 'nedelja', 'praznik', 'pocitnice']]
		# one-hot encoding za čase začetka
		x[df.index, df.cas + 6] = 1

		return x
	
	def train(self, x, y, alpha = 1):
		reg = skllm.Ridge(alpha, fit_intercept=True).fit(x, y)
		self.model = reg
	
	def predict_duration(self, x):
		return self.model.predict(x)
	
	def predict_arrival(self, x, start_t):
		return (start_t + pd.to_timedelta(self.predict_duration(x), unit='seconds'))




if __name__ == '__main__':
	filename = "./train_pred.csv"
	x, y, _ = ModelPred.open_train(filename)

	'''test_size = 1000
	ch = np.random.choice(range(len(y)), size=(test_size,), replace=False)    
	ind_test = np.zeros((len(y)), dtype=bool)
	ind_test[ch] = True
	ind_train = ~ind_test'''


	model = ModelPred()
	model.train(x, y, 1)

	filename_test = "./test_pred.csv"
	xt, start_t, _ = ModelPred.open_test(filename_test)

	predictions = model.predict_arrival(xt, start_t)
	predictions.to_csv('predtekmovanje.txt', header=False, sep=',', index=False, mode='wt', date_format=lpputils.FORMAT, quoting=csv.QUOTE_NONE)
