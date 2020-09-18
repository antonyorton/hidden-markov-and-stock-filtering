import numpy as np
import pandas as pd
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()  
import matplotlib.pyplot as plt
import hmm_datapreparation as dataprep
np.set_printoptions(precision=4) 


def forward_alg(obs,A,B,p,N,vocab):
	"""
		N: integer. Number of hidden states Si
		p: vector of length N. Initial state probabilities, pi = prob(q1 = Si)
		A: array of size  N x N. Transition matrix,        aij = prob(q(t+1) = Sj | q(t) = Si)
		B: array of size N x M. Observation probabilities. bij = prob(O(t) = vj | q(t) = Si)
		vocab: vector of length M. Set of all possible observations
		Obs: vector of any length. Observation sequence, each value must be part of the vocab.
	"""
	
	T = len(obs) #Total number of observations
	alpha = np.zeros((N,T)) #Forward probability matrix
	
	#zero step
	bcol = np.argmin(np.abs(vocab-obs[0])) #B column = f
	alpha[:,0] = p*B[:,bcol] #alpha i0 = pi*bif (componentwise)
	
	#recursive step through each time i
	for i in range(1,alpha.shape[1]):
		bcol = np.argmin(np.abs(vocab-obs[i])) #B column = f
		alpha[:,i] = np.dot(alpha[:,i-1],A) * B[:,bcol] #dot then componentwise
	
	prob_obs = np.sum(alpha[:,-1]) #probability of the observations given the model.
	
	return alpha, prob_obs
	
def backward_alg(obs,A,B,p,N,vocab):
	"""
		N: integer. Number of hidden states Si
		p: vector of length N. Initial state probabilities, pi = prob(q1 = Si)
		A: array of size  N x N. Transition matrix,        aij = prob(q(t+1) = Sj | q(t) = Si)
		B: array of size N x M. Observation probabilities. bij = prob(O(t) = vj | q(t) = Si)
		vocab: vector of length M. Set of all possible observations
		Obs: vector of any length. Observation sequence, each value must be part of the vocab.
	"""
	
	T = len(obs) #Total number of observations
	beta = np.zeros((N,T)) #Backward probability matrix
	
	#zero step
	beta[:,-1] = np.ones(N) #defined as all 1 since nothing after t=T
	
	#recursive step
	for i in np.arange(T-2,-1,-1):
		bcol = np.argmin(np.abs(vocab-obs[i+1])) #B column = f
		bvec = np.c_[B[:,bcol]*beta[:,i+1]]
		beta[:,i] = np.dot(A,bvec).T
		#beta[:,i] = np.dot(B[:,bcol]*beta[:,i+1],A.T) #componentwise then dot
		
	#final prob
	bcol = np.argmin(np.abs(vocab-obs[0])) #B column = f
	prob_obs = np.dot(p,beta[:,0]*B[:,bcol]) #componentwise then dot .. probability of the observations given the model.
	
	return beta, prob_obs
	
def baum_welch_alg(obs,A,B,p,N,vocab):
	
	"""
	N: integer. Number of hidden states Si
	p: vector of length N. Initial state probabilities, pi = prob(q1 = Si)
	A: array of size  N x N. Transition matrix,        aij = prob(q(t+1) = Sj | q(t) = Si)
	B: array of size N x M. Observation probabilities. bij = prob(O(t) = vj | q(t) = Si)
	vocab: vector of length M. Set of all possible observations
	Obs: vector of any length. Observation sequence, each value must be part of the vocab.
	"""
	
	fwd, prob = forward_alg(obs, A, B, p, N, vocab) #get forward algorithm results
	bwd, prob = backward_alg(obs, A, B, p, N, vocab) #get backward algorithm results
	
	gamma = fwd * bwd / prob #get gamma (which is just the componentwise multiplication of fwd and bwd divided by prob)
	
	#get xee matrices
	xee = np.zeros((len(obs)-1,A.shape[0],A.shape[1]))
	for t in range(len(obs)-1):
		alp = fwd[:,t]
		bet = bwd[:,t+1]
		bcol = np.argmin(np.abs(vocab-obs[t+1]))
		b = B[:,bcol]
		xee[t] = np.c_[alp] * A * b * bet / prob #xee (greek xi) matrix for time t
	
	############get updated p
	pnew = gamma[:,0]  #see Rabiner (1989) eqn 40a
	
	############get updated A
	Anew = np.sum(xee,axis=0) / np.c_[np.sum(np.sum(xee,axis=2),axis=0)]  #See Jurafsky and Martin (2019) Speech and Language processing Figure A.14
																		  # found here: https://web.stanford.edu/~jurafsky/slp3/A.pdf
	#######################
																		  
																		  
	##################get updated B																	  
	topsum = np.zeros_like(B)
	
	for m in range(len(vocab)):
		for t in range(len(obs)):
			if obs[t] == vocab[m]:
				topsum[:,m] += gamma[:,t] 	
			
	botsum = np.sum(gamma,axis=1)	
	Bnew = topsum / np.c_[botsum]  #See Jurafsky and Martin (2019) Speech and Language processing Figure A.14
									# found here: https://web.stanford.edu/~jurafsky/slp3/A.pdf
	#####################3
	
	#make a minimum value of 1e-5 (apparently these cause errors, see Rabiner (1989) Fig 17)
	pnew[pnew<1e-5]=1e-5
	Anew[Anew<1e-5]=1e-5
	Bnew[Bnew<1e-5]=1e-5
	#then normalise
	pnew = pnew/np.sum(pnew)
	Anew = Anew/np.c_[np.sum(Anew,axis=1)]
	Bnew = Bnew/np.c_[np.sum(Bnew,axis=1)]
	
	
	#get new prob
	fwd, prob = forward_alg(obs, Anew, Bnew, p, N, vocab)
	
	#print('P(obs.|model) = ',prob)
	
	return Anew, Bnew, pnew, prob
	
def viterbi_alg(obs,A,B,p,N,vocab):

	"""
	N: integer. Number of hidden states Si
	p: vector of length N. Initial state probabilities, pi = prob(q1 = Si)
	A: array of size  N x N. Transition matrix,        aij = prob(q(t+1) = Sj | q(t) = Si)
	B: array of size N x M. Observation probabilities. bij = prob(O(t) = vj | q(t) = Si)
	vocab: vector of length M. Set of all possible observations
	Obs: vector of any length. Observation sequence, each value must be part of the vocab.
	"""
	
	T = len(obs) #Total number of observations
	viterb = np.zeros((N,T)) #Viterbi probability matrix
	backptr = np.zeros((N,T)) #Backpointer matrix
	
	#zero step
	bcol = np.argmin(np.abs(vocab-obs[0])) #B column = f
	viterb[:,0] = p*B[:,bcol] #vit i0 = pi*bif (componentwise)
	backptr[:,0] = np.zeros(N) 
	
	#recursive step through each time i
	for i in range(1,viterb.shape[1]):
		bcol = np.argmin(np.abs(vocab-obs[i])) #B column = f
		rmat = viterb[:,i-1]*A*np.c_[B[:,bcol]] #temp matrix to extract results
		viterb[:,i] = np.max(rmat,axis = 1)
		backptr[:,i] = np.argmax(rmat,axis = 1)
		
	best_prob = np.max(viterb[:,-1]) #highest probability path.
	final_state = np.argmax(viterb[:,-1]) #state at end of path
	
	#get the best state path
	best_state_path = np.zeros(T,dtype=np.int)
	best_state_path[-1] = final_state
	for i in np.arange(T-1)[-1::-1]:
		best_state_path[i] = backptr[best_state_path[i+1],i+1]
	
	#get optimal B matrix
	bs = best_state_path[:]
	Bnew = np.zeros_like(B)
	for i in range(N):
		for j in range(len(vocab)):
			if len(obs[bs==i]) == 0:
				Bnew[i,j]=0
			else:
				Bnew[i,j] = np.sum(obs[bs==i]==vocab[j])/len(obs[bs==i]) #num times in state i and obs = vocab(j) / num times in state i
	


	Bnew[Bnew<1e-5]=1e-5
	Bnew = Bnew/np.c_[np.sum(Bnew,axis=1)]
	
	return Bnew, best_state_path

def OLDfit_hmm_model(obs,A,B,p,N,vocab,tolerance=1e-5):
	
	"""
	N: integer. Number of hidden states Si
	p: vector of length N. Initial state probabilities, pi = prob(q1 = Si)
	A: array of size  N x N. Transition matrix,        aij = prob(q(t+1) = Sj | q(t) = Si)
	B: array of size N x M. Observation probabilities. bij = prob(O(t) = vj | q(t) = Si)
	vocab: vector of length M. Set of all possible observations
	Obs: vector of any length. Observation sequence, each value must be part of the vocab.
	"""
	obs1 = obs[:]
	Anew = A[:]
	Bnew = B[:]
	pnew = p[:]
	
	#print('fitting ..')
	fwd, prob_old = forward_alg(obs1, Anew, Bnew, p, N, vocab)
	

	prob = 0 #dummy initialisation
	diff = 0 #dummy initialisation
	
	while (np.abs(diff-1) > tolerance):
		Anew, Bnew, pnew, prob = baum_welch_alg(obs1, Anew,Bnew, pnew, N, vocab)
		diff = prob/prob_old
		prob_old = prob
		#print(prob)
		
	#print('P(obs.|model) = ',prob)
	return Anew, Bnew, pnew, prob
	
def new_fit_hmm_model(data,A,B,p,N,vocab,data_n_range = [2520,2620],num_back_windows = 30, tolerance=1e-5):
	
	"""
	N: integer. Number of hidden states Si
	p: vector of length N. Initial state probabilities, pi = prob(q1 = Si)
	A: array of size  N x N. Transition matrix,        aij = prob(q(t+1) = Sj | q(t) = Si)
	B: array of size N x M. Observation probabilities. bij = prob(O(t) = vj | q(t) = Si)
	vocab: vector of length M. Set of all possible observations
	data: dataFrame extracted from dataprep.get_stock_data() function, ie the observations
	data_n_range: list index [min,max] of data window to fit
	num_back_windows: number of back windows to use in fitting process
	"""
	
	d0 = data_n_range[0]
	d1 = data_n_range[1]
	latency = d1-d0
	obswindows = []
	for i in np.arange(num_back_windows,0,-1):
		obswindows.append(dataprep.get_sample_from_data(data,d0-i,latency)['gain_group'].values)
	
	Anew = A[:]
	Bnew = B[:]
	pnew = p[:]
	
	#obs1 = obs[:]
	
	#print('fitting ..')
	
	#fit through each of the back obs windows
	for i in range(len(obswindows)):
		obs1 = obswindows[i]
		Anew, Bnew, pnew, prob = baum_welch_alg(obs1, Anew,Bnew, pnew, N, vocab)
		#print('prob backwin = ',prob)
	
	#main window to fit to
	obs1 = dataprep.get_sample_from_data(data,d0,latency)['gain_group'].values

	fwd, prob_old = forward_alg(obs1, Anew, Bnew, p, N, vocab)
	
	prob = 0 #dummy initialisation
	diff = 0 #dummy initialisation
	
	#then continue fitting until convergence
	while (np.abs(diff-1) > tolerance):
		Anew, Bnew, pnew, prob = baum_welch_alg(obs1, Anew,Bnew, pnew, N, vocab)
		diff = prob/prob_old
		prob_old = prob
		#print(prob)
		
	#print('P(obs.|model) = ',prob)
	return Anew, Bnew, pnew, prob
	
	
#early experiment only
def find_an_initial_p(obs,A,B,p,N,vocab,num_check = 5):
	
	"""fits model for a number of randomly selected p's and returns the one with the highest probability
		against the observations
	"""
	
	p_all = np.zeros((num_check,N))
	probs_all = np.zeros(num_check)
	for i in range(num_check):
		p = np.random.rand(N) #random vector length N
		p/=np.sum(p) #normalise
		#print('p = ',p)
		A1,B1,p1,prob = fit_hmm_model(obs,A,B,p,N,vocab,tolerance=1e-5)
		p_all[i,:] = p
		probs_all[i] = prob

	p = p_all[np.argmax(probs_all)] #choose best p
	#print('best p is ',p)
	return p

	
if __name__ == "__main__":

	#sample length
	start_num = 1000 #starting observation number
	N=4 #num states
	
	if 1==1:
	
		sampN = 100 #train set length
		predN = 50	#predict set length

		#get sample daily observation set from market
		data, percents, vocab = dataprep.get_stock_data(ticker = '^AXJO', startdate='2009-03-09', enddate='2020-07-25', check_data = False)
		obs = dataprep.get_sample_from_data(data,start_num,sampN + predN) #extract a consecutive sample
		obs = obs['gain_group'].values  #convert to vocabulary only (ie enumerated percentage bins, -7 most negative to 7 most positive)
		
		obs_train = obs[0:sampN]
		obs_test = obs[sampN:sampN+predN]
		
		M = len(vocab) #num vocab
		
		#initial params
		
		#B = np.ones((N,M))*1/N  #size = num states x num vocab
		#A = np.ones((N,N))*1/N  #size = num states x num states
		
		#random start params
		A = np.random.rand(N,N)	
		A = A / np.c_[np.sum(A,axis=1)]
		B = np.random.rand(N,M)	
		B = B / np.c_[np.sum(B,axis=1)]			
		p = np.random.rand(N)
		p /= np.sum(p)
		
		#get a better B using the viterbi algorithm
		B, bsp = viterbi_alg(obs,A,B,p,N,vocab)
		
		
		#Get a reasonable starting p using random values and selecting the best
		#p = find_an_initial_p(obs_train,A,B,p,N,vocab,num_check = 10)
		
		#fit model
		A1,B1,p1,prob = fit_hmm_model(obs_train,A,B,p,N,vocab,tolerance=1e-5) #fit model
		

	
	#Test problem
	#obs = np.array([3,1,3])
	#vocab = np.array([1,2,3])
	#B = np.array([[0.2,0.1,0.4],[0.5,0.1,0.1]])
	#A = np.array([[0.6,0.4],[0.5,0.5]])
	#N = 2
	#p = np.array([0.8,0.2])