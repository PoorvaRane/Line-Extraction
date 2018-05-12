import numpy as np
import sklearn as sk
import matplotlib
import matplotlib.pyplot as plt
import scipy.stats
from pygam import LinearGAM
from pygam.utils import generate_X_grid
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import roc_auc_score, accuracy_score
import pandas as pd
from scipy.special import logit, expit
import math
import sys


def which_robust_min(df):
    alpha = 0.1
    pred = df['predicted']
    cutoff = np.min(pred) + (np.max(pred) - np.min(pred))*alpha
    return int(np.median(df['idx'][pred<=cutoff]))


def main():
	csv_file = sys.argv[1]
	dat = pd.read_csv(csv_file)
	#For now, drop the top and bottom 5% of the image before we model it
	dat.columns = ['y']
	dat['idx'] = range(dat.shape[0])
	alpha = .05
	#Subset of data.
	datsub = dat.iloc[int(np.floor(alpha*dat.shape[0])):int(np.ceil((1-alpha)*dat.shape[0])),]

	#I'll search over line heights from 20 to 100.
	lb =20
	ub = 120

	#For each possible line shift, we model the number of black pixels as
	#Gaussian with a smooth, periodic mean with period equal to the line shift.
	#For our first search, we'll actually use a Gaussian model because it's much 
	#faster and doesn't really matter.  We look for the fit which minimizes the deviance/RSS.
	dlist = range(lb, ub+1)
	devlist = np.zeros(len(dlist))
	for i in range(len(dlist)):
	    stride = dlist[i]
	    datsub.loc[:,'looped'] = (datsub.loc[:,'idx'] % (stride))
	    fitted = LinearGAM().fit(datsub.loc[:,'looped'],datsub['y'])
	    devlist[i] = fitted.statistics_['loglikelihood']

	# Uncomment to plot
	# plt.plot(dlist, devlist)
	# plt.show()

	stride = dlist[np.argmax(devlist)]

	#For each possible line shift, we model the number of black pixels as
	#Gaussian with a smooth, periodic mean with period equal to the line shift.
	#For our first search, we'll actually use a Gaussian model because it's much 
	#faster and doesn't really matter.  We look for the fit which minimizes the deviance/RSS.
	dlist = np.linspace(stride-1, stride+1, 100)
	# print(dlist)
	devlist = np.zeros(len(dlist))
	for i in range(len(dlist)):
	    stride = dlist[i]
	    datsub.loc[:,'looped'] = (datsub.loc[:,'idx'] % (stride))
	    fitted = LinearGAM().fit(datsub.loc[:,'looped'],datsub['y'])
	    devlist[i] = fitted.statistics_['loglikelihood']

	# Uncomment to plot
	# plt.plot(dlist, devlist)
	# plt.show()
	stride = dlist[np.argmax(devlist)]

	datsub.loc[:,'looped'] = (datsub.loc[:,'idx'] % (stride))
	fitted = LinearGAM().fit(datsub.loc[:,'looped'],datsub['y'])

	dat.loc[:,'looped'] = (dat.loc[:,'idx'] % (stride))
	yhat = fitted.predict(dat.loc[:,'looped'])
	dat.loc[:,'predicted'] = yhat
	dat.loc[:,'block'] = (dat.loc[:,'idx']- dat.loc[:,'looped'])/stride

	# # Uncomment to plot
	# plt.scatter(dat['looped'],dat['y'])
	# plt.plot(dat['looped'], dat['predicted'], color='red')
	# plt.show()
	# #Ignore the weird red thing at the bottom, that's just because of the periodicity and connecting the line.  
	# #Seems to work though!

	ret = dat.groupby('block').aggregate(which_robust_min).iloc[:,1]
	#which_robust_min(dat.loc[dat['block']==4.0])

	dat['y'].plot()
	for i in range(ret.shape[0]):
	    plt.axvline(x=ret.iloc[i], color='red')
	# # Uncomment to plot
	# plt.show()

	cuts = []
	search_list = np.arange(int(math.floor(stride/(3)))*-1, int(math.floor(stride/(3))) + 1)
	window_width = int(stride*1)
	window = np.arange(-int(window_width), int(window_width) + 1)
	minidx = ret.values

	for i in range(len(minidx)):
	    idx = minidx[i]
	    if (idx + max(window) >= dat.shape[0]):
	        continue
	    # RSSs is a measure of the goodness of the fit for every epsilon
	    RSSs = np.zeros(len(search_list))
	    
	    for j in range(len(search_list)):
	        idxs = np.arange(idx, idx + int(window_width)+1)
	        if (idx + search_list[j] < 0):
	            RSSs[j] = sys.maxsize
	            continue
	        # reduce difference between data wiggle and orginal fit
	        if (max(idxs) + search_list[j] >= dat.shape[0]):
	            continue
	        RSSs[j] = sum(np.square(dat.y.values[idxs + search_list[j]] - dat.predicted.values[idxs]))
	    
	    temp_sum = minidx[i] + search_list[np.argmin(RSSs) - 1]
	    cuts.append(temp_sum)
	    cuts.append(temp_sum + int(math.floor(stride)))
	    
	cuts_np = np.array(cuts)
	cuts_np = cuts_np.reshape(cuts_np.shape[0]/2, 2)
	cuts_np = np.array(cuts_np, dtype=int)

	fname = sys.argv[1].split('.')[0]+'.csv'
	np.savetxt(fname, cuts_np, delimiter=",", fmt='%i')


if __name__ == '__main__':
	main()