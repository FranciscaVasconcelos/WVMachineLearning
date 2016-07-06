# runs best machine learning model on data acquired by RaspberryPi

# imports
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import numpy as np
from scipy import stats
from sklearn import svm
from sklearn.externals import joblib
import pickle
import json

# acquires number of bins
with open('parameters.json') as data_file:
    dic = json.load(data_file)
bin_num = dic['num_bins']

# get spectra data from RaspberryPi
tmpdata = np.genfromtxt('grapes_white_transpose.csv', delimiter=',')
test = np.nan_to_num(tmpdata)

# bin the data
test_data,bin_edges_ts,binnum_ts=stats.binned_statistic(range(test.shape[1]), test, 'median', bins=int(bin_num))

# load model from pkl file
a_result=joblib.load("model.pkl")

# apply model to RaspberryPi spectra data
for i in test_data:
    print(a_result.predict(i.reshape(1,-1)))

