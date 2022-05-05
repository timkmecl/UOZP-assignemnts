import numpy as np
from scipy.sparse import csr_matrix
import gzip
import csv
import lpputils
import datetime
import matplotlib.pyplot as plt
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
import pandas as pd


class Model():
	def open_train(self, filename):
		df = pd.read_csv(filename, sep="\t")
		df = df[['Route Direction', 'Departure time', 'Arrival time']]
		df = df.rename(columns = {'Route Direction':'route', 'Departure time':'start', 'Arrival time':'end'})
		
		df.start = pd.to_datetime(df.start, format=lpputils.FORMAT)
		df.end = pd.to_datetime(df.end, format=lpputils.FORMAT)
		
		self.routes = df.route.unique()

		y  = ((df.end - df.start)/pd.Timedelta(seconds=1)).astype(int)
		x = self.transform_data(df)


		return x, y

	def open_test(self, filename):
		df = pd.read_csv(filename, sep="\t")
		df = df[['Route Direction', 'Departure time']]
		df = df.rename(columns = {'Route Direction':'route', 'Departure time':'start'})
		
		df.start = pd.to_datetime(df.start, format=lpputils.FORMAT)
		
		x = self.transform_data(df)

		return x, df.start
	
	def transform_data(self, df):
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

		# minuta začetka vožnje
		df['cas'] = df.start.dt.hour*60 + df.start.dt.minute

		df['pon_cet'] = df.dan_v_tednu < 4
		df['petek'] = df.dan_v_tednu == 4
		df['sobota'] = df.dan_v_tednu == 5
		df['nedelja'] = df.dan_v_tednu == 6
		df['praznik'] = df.dayofyear.isin(PRAZNIKI)
		# štetje mesecev začne poleti (z junijem)
		df['mesec'] = (df.start.dt.month + 6) % 12

		# preveri, ali pade dan v letu znotraj intervala katerih od počitnic
		df['pocitnice'] = False
		for i, d in POCITNICE.iterrows():
			df.pocitnice = df.pocitnice | ((d.start <= df.dayofyear) & (d.end >= df.dayofyear))
		
		
		x = np.zeros((len(df), 9))
		x[:, 0:8] = df[['mesec', 'cas', 'pon_cet', 'petek', 'sobota', 'nedelja', 'praznik', 'pocitnice']]
		for i, route in enumerate(self.routes):
			x[df['route']==route, 8] = i

		return x
	
	def train(self, x, y):
		reg = HistGradientBoostingRegressor(min_samples_leaf=5, max_iter=500, categorical_features=[2,3,4,5,6,7,8]).fit(x, y)
		self.model = reg
	
	def predict_duration(self, x):
		return self.model.predict(x)
	
	def predict_arrival(self, x, start_t):
		return (start_t + pd.to_timedelta(self.predict_duration(x), unit='seconds'))


def ni_poleti(x):
	# upošteva zamik štetja mesecev
	return np.isin(x, (np.array([1,2,3,4,5, 9,10,11,12]) + 6) % 12)


if __name__ == '__main__':
	filename = "./train.csv"
	
	model = Model()
	x, y = model.open_train(filename)

	# za učenje ne uporabi poletnih mesecev
	ni_poletje = ni_poleti(x[:,0])
	x = x[ni_poletje , :]
	y = y[ni_poletje]

	'''
	# delitev na učne in testne podatke za preverjanje
	test_size = 100000
	ch = np.random.choice(range(len(y)), size=(test_size,), replace=False)    
	ind_test = np.zeros((len(y)), dtype=bool)
	ind_test[ch] = True
	ind_train = ~ind_test

	model.train(x[ind_train], y[ind_train])
	xt, start_t = model.open_test(filename)
	ni_poletje = ni_poleti(xt[:,0])
	xt = xt[ni_poletje , :]
	start_t = start_t[ni_poletje]

	yh = model.predict_duration(xt[ind_test])
	print(np.sum(np.abs(yh - y[ind_test]))/test_size)
	'''
	
	
	model.train(x, y)
	
	filename_test = "./test.csv"
	xt, start_t = model.open_test(filename_test)

	predictions = model.predict_arrival(xt, start_t)
	predictions.to_csv('tekmovanje.txt', header=False, sep=',', index=False, mode='wt', date_format=lpputils.FORMAT, quoting=csv.QUOTE_NONE)