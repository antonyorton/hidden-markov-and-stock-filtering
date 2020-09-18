import yfinance as yf
import numpy as np
import pandas as pd
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()  
import matplotlib.pyplot as plt
np.set_printoptions(precision=4) 

def get_stock_data(ticker = '^AXJO', startdate='2009-03-09', enddate='2020-07-06', check_data = False):
	"""downloads data from yahoo finance and creates bins with daily close divided by open percentage"""
	
	"""returns: 1. data, 2. bin percentage Close/Opens 2. enumerated bins """
	
	#download data
	data = yf.download(ticker, start = startdate, end = enddate)
	
	#get percent close/open
	daygain = np.zeros(len(data))
	daygain[1::] = data['Close'].values[1::]/data['Close'].values[0:-1]-1
	
	data['percent_CC'] = daygain

	#create input data groups for percentage O/C
	#a1 = np.array([0,0.002,0.004,0.006,0.008,0.01,0.012,0.014])
	a1 = np.array([0,0.001,0.002,0.003,0.004,0.005,0.006,0.007,0.008,0.009,0.01])
	groups = groups = np.hstack((-a1[-1:0:-1],a1)) #bins of different daily percentage open / close results

	#split percentage O/C into discrete bins
	data['gain_group'] = data['percent_CC'].apply(lambda x : np.argmin(np.abs(groups-x)))-(len(a1)-1)

	#check data - depreciated
	if check_data:
		fig, ax1 = plt.subplots(constrained_layout=True)
		ax2 = ax1.twinx()
		ax1.plot(pd.to_datetime(data.index),data['percent_CC'])
		ax1.set_ylim((-0.03,0.03))

		#ax2.plot(data.index,data['gain_group'],'or')
		#ax2.set_ylim((-10.5,10.5))
		ax2.plot(pd.to_datetime(data.index),data['gain_group']*0.02/7,'r')
		ax2.set_ylim((-0.03,0.03))
		plt.grid(True)
		plt.show()

	#returns data, bin percentages, enumerated bins
	return data, groups, np.arange(len(groups))-(len(a1)-1)

def get_sample_from_data(data, n_start,num):
	""" get a consecutive sample from data
	data: array like
	n_start: integer, index of start of sample
	num: integer, number of sample points
	returns data[n_start:n_start + num]
	"""
	
	if n_start+num > len(data):
		raise(ValueError)
		print('error: n_start + num > len(data)')
		
	if type(data) == pd.DataFrame:
		return data.iloc[n_start:n_start+num]
	elif type(data) == np.array:
		return data[n_start:n_start+num]
	else:
		raise(ValueError)
		print('unknown format for input data')
		
	return


	
if __name__ == "__main__":

	data, percents, vocab = get_stock_data(ticker = 'CBA.AX', startdate='2009-03-09', enddate='2020-07-06', check_data = False)
	
	obs = get_sample_from_data(data,2000,100) #extract a consecutive sample
	obs = obs['gain_group'].values  #convert to vocabulary only (ie enumerated percentage bins, -7 most negative to 7 most positive)
	
	#NEXT: need to set up and test forward and backward algorithms on obs, given model parameters {p0, A, B}
	
	#THEN: set up Baum-Welch algorithm and learn the HMM parameters based on many different samples