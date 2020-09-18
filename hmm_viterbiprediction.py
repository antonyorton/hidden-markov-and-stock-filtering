import numpy as np
import pandas as pd
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()  
import matplotlib.pyplot as plt
import hmm_datapreparation as dataprep
import hmm_algorithms as hma
np.set_printoptions(precision=4)
from importlib import reload
reload(hma)
reload(dataprep)

def get_future(model,bsp):
	
	A,B,p,N,vocab = model
	
	last_state = bsp[-1]
	
	next_vocab_predict = np.c_[A[last_state,:]] * B
	next_vocab_predict = np.sum(next_vocab_predict,axis = 0)
	next_vocab_predict = vocab[np.argmax(next_vocab_predict)]
	
	return next_vocab_predict

if __name__ == "__main__":

	#get sample daily observation set from market
	data, percents, vocab = dataprep.get_stock_data(ticker = '^AXJO', startdate='1988-03-09', enddate='2020-08-08', check_data = False)
				
	#sample length

	start_num = 60 #starting prediction observation number
	Nvals= [5] #num states
	latency = 40 #length of training set
	predN = 6940 #length of prediction set
	
	N = Nvals[0]
	
	obs = dataprep.get_sample_from_data(data,start_num - latency,latency + predN) #extract a consecutive sample from the data
	obs = obs['gain_group'].values  #convert to vocabulary only (ie enumerated percentage bins, -7 most negative to 7 most positive)

	
	num_runs = 7 #num of runs to check for each prediction (guess is based on number of predicted short vs predicted long)
	total_vocab_sum = 0 #total gain loss if bet using hmm
	total_market_sum = 0 #total gain loss if do nothing
	record = np.zeros((predN,4))
	
	for kk in np.arange(predN):
		
		start_num=start_num+1
		print('predicting day number = ',start_num,'  ###########################################' )
		
		obs_train = obs[0+kk:latency+kk]
		obs_test = obs[latency+kk:latency+kk+1]
		
		M = len(vocab) #num vocab
		
		if 1==2:
			guess = []
			for runs in np.arange(num_runs): 
				#random start params DO NOT DELETE (seems to be that a random start param set is required)
				A = np.random.rand(N,N)	
				A = A / np.c_[np.sum(A,axis=1)]
				B = np.random.rand(N,M)	
				B = B / np.c_[np.sum(B,axis=1)]			
				p = np.random.rand(N)
				p /= np.sum(p)
				
				#non-random start params DO NOT DELETE
				#A = np.ones((N,N))*(1/N)
				#B = np.ones((N,M))*(1/M)
				p = np.ones(N)*(1/N)
				
				#get a better B using the viterbi algorithm
				B, bsp = hma.viterbi_alg(obs_train,A,B,p,N,vocab)
				
				#print('initial best state path')
				#print(bsp)
				#print(A)

				#fit model
				#for i in range(20):
				#A,B,p,prob = hma.fit_hmm_model(obs_train,A,B,p,N,vocab,tolerance=1e-5) #fit model
				A,B,p,prob = hma.new_fit_hmm_model(data,A,B,p,N,vocab,data_n_range=[start_num,start_num+latency],num_back_windows = int(0.8*latency),tolerance=1e-5) #fit model

				
				model = [A, B, p, N, vocab]
				#get best state path
				Btemp, bsp = hma.viterbi_alg(obs_train,A,B,p,N,vocab)

				
				#print('fitted best state path')
				future_guess = get_future(model,bsp)
				guess.append(future_guess)
				#print('predict = ',future_guess,'   actual = ',obs_test[0])
			
			guess = np.array(guess)
		
		guess = np.random.rand(21)-0.5
		
		if len(guess[guess<0]) > len(guess[guess>0]): #more negatives - SHORT
			guess = -1
		elif len(guess[guess<0]) < len(guess[guess>0]): #more positives - LONG
			guess = 1
		else:
			guess = 0 #no winner - NO BET
			
		

		
		
		cost_vocab = 2.15 #cost per day of trading (in vocab terms)
		if guess >=0: #LONG BET - HEDGED
		#	print('WIN OF = ',guess * obs_test[0], '    MARKET GAIN OF = ',obs_test[0])
		#	total_vocab_sum += guess * obs_test[0]
			print('WIN OF = ',np.round(np.max([-cost_vocab,guess * obs_test[0]-cost_vocab]),2), '    MARKET GAIN OF = ',obs_test[0])
			total_vocab_sum += np.round(np.max([-cost_vocab,guess * obs_test[0]-cost_vocab]),2)
		else:	#SHORT BET - HEDGED AT 2 cost (assumes vocab = 1 signifies 0.1%, ie so 0.2% hedge cost)
			print('WIN OF = ',np.round(np.max([-cost_vocab,guess * obs_test[0]-cost_vocab]),2), '    MARKET GAIN OF = ',obs_test[0])
			total_vocab_sum += np.round(np.max([-cost_vocab,guess * obs_test[0]-cost_vocab]),2)
		
		total_market_sum += obs_test[0]
		print('TOTAL = ',np.round(total_vocab_sum,2), '                 TOTAL MARKET = ',total_market_sum)
		record[kk] = np.array([data.index[start_num].to_julian_date(),guess,total_vocab_sum,total_market_sum])
		

		
	#plt.plot(record[:,2],'r',label = 'predict')
	#plt.plot(record[:,3],'b',label = 'market')
	#plt.legend()
	#plt.show()
	#data  = pd.DataFrame(record,columns=['date','SHORT/LONG','predict','market'])
	#data.to_csv('results_hedged_1988to2020_doublehedge_N5.csv',index=False)