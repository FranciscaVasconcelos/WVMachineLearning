from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import numpy as np
from scipy import stats
from sklearn import svm
from sklearn.externals import joblib
import json

#MODULO 1
#LABELS SEPARATED FROM THE MATRIX

#loading data
tmpdata = np.genfromtxt('np_specg.csv', delimiter=',')
X = np.nan_to_num(tmpdata)

# Creation of labels
y = []
for i in range(0,27):
    y.append(1)
for i in range(27,53):
    y.append(2)

with open('parameters.json') as data_file:
    dic = json.load(data_file)

model = dic['model']
print(model)
bin_num = dic['num_bins']
parameter = dic['parameter']

#MODULO2
#PROCESS THE DATA(BINNING)

#binning model matrix,binned matrix==stat 
stat, bin_edges, binnum = stats.binned_statistic(range(X.shape[1]), X, 'median', bins=int(bin_num))
print(stat.shape)

#MODULO3
#APPLY THE MODEL AND PRINT THE RESULT

if model == 'svm.LinearSVC()':
    clf=svm.LinearSVC(C=parameter)
if model == 'RandomForestClassifier()': 
    clf=RandomForestClassifier(n_estimators=parameters,n_jobs=-1)
if model == 'LinearDiscriminantAnlysis()': 
    clf=LinearDiscriminantAnalysis()

clf.fit(stat,y)
print(clf)
joblib.dump(clf ,"model.pkl")
#a_result=joblib.load("model.pkl")

