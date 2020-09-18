import numpy as np
import pandas as pd
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()  
import matplotlib.pyplot as plt
import hmm_datapreparation as dataprep
np.set_printoptions(precision=4)
from scipy import optimize

def get_exp_moving_avg(data,n_steps = 5):
	if type(data) == np.ndarray:
		try:
			data1 = pd.Series(data)
		except:
			raise ValueError('array data must be 1 dimensional only')
	elif type(data) == pd.DataFrame:
		data1 = data[list(data)[0]].copy()
	elif type(data) == pd.Series:
		data1 = data.copy()
	else:
		raise ValueError('data must be array or pandas series or dataframe')

	#print(data1.head)
	ema = data1.ewm(span=n_steps,adjust = False).mean()
	return ema

def plot_tickers(data, tickers = ['MQG'], mavgs = 'none',mavgs_colour = 'r',plot_show = True):
	"""data input is csv file with one 'Date' column and one column for each item of ticker (list)
		mavgs = 'none', 'traders' ([3,5,8,10,15,20]), 'investors'([35,40,45,50,60]) or 'both'
	"""
	
	if not mavgs in ['none','traders','investors','both']:
		raise ValueError('Error: mavgs must be either "none", "traders", "investors" or "both"')
	
	data['Date'] = pd.to_datetime(data['Date'])
	
	if mavgs == 'both':
		trader_mavgs = [3,5,8,10,15,20]
		investor_mavgs = [35,40,45,50,60]
	
		for i in range(len(tickers)):
			plt.plot_date(data['Date'],data[tickers[i]], 'k',linewidth = 1.0,zorder=10,label = tickers[i])

			for j in range(len(trader_mavgs)):
				if j==0: #just to add legend to one of the lines
					plt.plot_date(data['Date'],get_exp_moving_avg(data[tickers[i]],n_steps = trader_mavgs[j]),'r',linewidth = 1.0,zorder=5,label = 'Traders')
				else:
					plt.plot_date(data['Date'],get_exp_moving_avg(data[tickers[i]],n_steps = trader_mavgs[j]),'r',linewidth = 1.0,zorder=5)
				
					
			for j in range(len(investor_mavgs)):
				if j==0:
					plt.plot_date(data['Date'],get_exp_moving_avg(data[tickers[i]],n_steps = investor_mavgs[j]),'b',linewidth = 1.0,zorder=0,label = 'Investors')
				else:
					plt.plot_date(data['Date'],get_exp_moving_avg(data[tickers[i]],n_steps = investor_mavgs[j]),'b',linewidth = 1.0,zorder=0)

	else:
		if mavgs == 'none':
			mavgs = []
		elif mavgs == 'traders':
			mavgs = [3,5,8,10,15,20]
		elif mavgs == 'investors':
			mavgs = [35,40,45,50,60]
		
		for i in range(len(tickers)):
			plt.plot_date(data['Date'],data[tickers[i]], 'k',linewidth = 1.0,zorder=0,label = tickers[i])
			#if len(mavgs)>0:
			for j in range(len(mavgs)):
				plt.plot_date(data['Date'],get_exp_moving_avg(data[tickers[i]],n_steps = mavgs[j]),mavgs_colour,linewidth = 1.0,zorder=0)
		
	plt.grid(True)
	plt.ylabel('Closing price $AUD')
	plt.legend()
	if plot_show:
		plt.show()

def get_movingavgs(data,ticker = 'MQG'):
	"""data input is csv file with one 'Date' column and one column named ticker
	Returns two 1D arrays: (1. Guppy traders moving averages = [3,5,8,10,15,20], 2. Guppy investors moving averages = [35,40,45,50,60])
	"""
	datanew = data[ticker]
	
	traders = [3,5,8,10,15,20]
	investors = [35,40,45,50,60]
	
	trader_ma = np.c_[get_exp_moving_avg(datanew,n_steps = traders[0])]
	for i in range(1,len(traders)):
		trader_ma = np.hstack((trader_ma,np.c_[get_exp_moving_avg(datanew,n_steps = traders[i])]))
	
	investor_ma = np.c_[get_exp_moving_avg(datanew,n_steps = investors[0])]
	for i in range(1,len(investors)):
		investor_ma = np.hstack((investor_ma,np.c_[get_exp_moving_avg(datanew,n_steps = investors[i])]))
		
	return (trader_ma,investor_ma)
	
if __name__ == "__main__":


	#interesting stocks 'FZO.AX', 'CDA.AX', 'MP1(number 1)','SAR','A2M', 'VOR (clear)'
	start = '1996-09-05'
	end = '2020-09-05'
	tot_days = (pd.to_datetime(end) - pd.to_datetime(start)).days
	all_tickers = pd.read_csv('ASXListedCompanies.csv',skiprows=1)
	all_tickers['ASX code'] = all_tickers['ASX code'].apply(lambda x:x+'.AX')
	#all_tickers = pd.read_csv('market_beaters.csv')
	rand1 = int(np.random.rand()*len(all_tickers))
	ticker = all_tickers.iloc[rand1]['ASX code']
	company = all_tickers.iloc[rand1]['Company name']
	m_avgsT = [3,5,8,10,15,20] #Traders moving average periods
	m_avgsI = [35,40,45,50,60] #Traders moving average periods
	timestep=1  #1 for daily, 5 for weekly etc
	
	
	
	#get indicies of all those stocks higher at date end compared to date start
	if 1==2:
		inds = []
		for i in range(len(all_tickers)):
			try:
				data, percents, vocab = dataprep.get_stock_data(ticker = all_tickers.iloc[i]['ASX code'], startdate=start, enddate=end, check_data = False)
				tempdat = data['Low'].rolling(window = int(tot_days / 30)).mean()
				if len(tempdat)>int(tot_days / 30):
					if tempdat[-1] > tempdat[int(tot_days / 30)] * tot_days / 365 * 1.06: #filter to stocks that have gone up at least 6% each year
						inds.append(i)
						print(i)
			except:
				print('Error: Unable to load data','ticker no ',i)
	
	#get all stocks between start and end date
	print(len(all_tickers))
	if 1==2:
		inds = []
		data, percents, vocab = dataprep.get_stock_data(ticker = 'BHP.AX', startdate=start, enddate=end, check_data = False) #get dates only
		alldat = pd.DataFrame(index=data.index)
		for i in range(len(all_tickers)):
			try:
				data, percents, vocab = dataprep.get_stock_data(ticker = all_tickers.iloc[i]['ASX code'], startdate=start, enddate=end, check_data = False)
				alldat[all_tickers.iloc[i]['ASX code'][0:-3]]=data['Close']
				print(i)
			except:
				print('Error: Unable to load data','ticker no ',i)
		alldat.to_csv('all_stocks_close_1996to2020.csv')

		
	#read all stocks, get guppys and apply filters
	#2020-09-08 to be continued ..
	
	
	#get the well performing stocks and save to csv (To be updated)
	if 1==2:
		data, percents, vocab = dataprep.get_stock_data(ticker = 'BHP.AX', startdate=start, enddate=end, check_data = False) #get dates only
		gooddat = pd.DataFrame(index=data.index)
		for i in range(len(all_tickers)):
			print(i)
			try:
				data, percents, vocab = dataprep.get_stock_data(ticker = all_tickers.iloc[i]['ASX code'], startdate=start, enddate=end, check_data = False)
	
				gooddat[all_tickers.iloc[i]['ASX code'][0:-3]]=data['Close']

			except:
				print('Error: Unable to load data','ticker no ',i)
		gooddat.to_csv('good_stocks_close.csv')#,index=False)	
	
	
	#plot some random stocks with guuppy MAs
	if 1==2:
		gooddat = pd.read_csv('good_stocks_close.csv')
		
		#company = all_tickers.iloc[i]['Company name']
		#dates = pd.to_datetime(gooddat['Date']) #dates
		dates = np.arange(len(gooddat)) # use numbers instead of dates to avoid weekend bumps
		tot_days = len(dates) #total days
		x=np.arange(tot_days)
		
		#mylist = ['CDA', 'MP1','SAR','A2M', 'VOR']
		mylist = list(gooddat)[1::]
		for i in range(len(mylist)):
			ticker = mylist[i]
			y = gooddat[ticker].values #closing prices
			
			if 5==5:
				plt.plot(dates[0::timestep], gooddat[ticker].values[0::timestep], ".g") #stock price
				#plt.plot(x[0::timestep], data['Close'].values[0::timestep], ".r")
				for jj in range(len(m_avgsT)):
					#mvdata = gooddat[ticker].rolling(window = jj).mean()[0::timestep]
					mvdata = get_exp_moving_avg(gooddat[ticker],n_steps = m_avgsT[jj])
					plt.plot(dates[0::timestep],mvdata,'r')
				for jj in range(len(m_avgsI)):
					#mvdata = gooddat[ticker].rolling(window = jj).mean()[0::timestep]
					mvdata = get_exp_moving_avg(gooddat[ticker],n_steps = m_avgsI[jj])
					plt.plot(dates[0::timestep],mvdata,'b')
				plt.title(ticker+' -- '+start+' to '+end)	
				plt.show()
						
		#tck = interpolate.splrep(x, y, k=2, s=1)
		#plt.plot(x,y,'r')
		#plt.plot_date(x,tck[1][0:-2],'b')
		#plt.show()