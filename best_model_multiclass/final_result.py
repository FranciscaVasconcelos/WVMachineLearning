# Takes model selected by search_model_multi and makes it easily accessible to the RaspberryPi through a pkl file

# imports
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import numpy as np
from scipy import stats
from sklearn import svm
from sklearn.externals import joblib
import json

# Load the spectra data
tmpdata = np.genfromtxt('grapes_white_transpose.csv', delimiter=',')
X = np.nan_to_num(tmpdata)


# Load the spectra labels
tmpdata = np.genfromtxt('labels.csv', delimiter=',')
y = np.nan_to_num(tmpdata)

# Load the model parameters
with open('parameters.json') as data_file:
    dic = json.load(data_file)
model = dic['model']
bin_num = dic['num_bins']
parameter = dic['parameter']

# bin the data 
stat, bin_edges, binnum = stats.binned_statistic(range(X.shape[1]), X, 'median', bins=int(bin_num))

# prepare the choosen model to be saved in the pkl file
if model == 'svm.LinearSVC()':
    clf=svm.LinearSVC(C=parameter)
if model == 'RandomForestClassifier()': 
    clf=RandomForestClassifier(n_estimators=parameters,n_jobs=-1)
if model == 'LinearDiscriminantAnlysis()': 
    clf=LinearDiscriminantAnalysis()

# save model to pkl file
clf.fit(stat,y)
joblib.dump(clf ,"model.pkl")

