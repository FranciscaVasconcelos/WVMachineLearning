from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
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
    data = json.load(data_file)

print(data)


#implement the label in the matrix(x_tr)
#x_tr = np.concatenate((x_tr, y_tr.reshape((-1,1))), axis=1)

# load test

test_data= np.genfromtxt("test",dtype=np.float)

#MODULO2
#PROCESS THE DATA(BINNING)
bin_num=130

#binning model matrix,binned matrix==stat 
stat, bin_edges, binnum = stats.binned_statistic(range(X.shape[1]), X, 'median', bins=int(bin_num))

#binning test spectrum,binned spectrum==test
test,bin_edges_ts,binnum_ts=stats.binned_statistic(range(X.shape[1]), test_data, 'median', bins=int(bin_num))

#MODULO3
#APPLY THE MODEL AND PRINT THE RESULT



#if the chosen model is "SupportVectorMachine", save the prediction

clf=svm.LinearSVC(C=int(10**3))
clf.fit(stat,y)
clf.predict(test.reshape(1,-1))
joblib.dump(clf ,"model.pkl")
a_result=joblib.load("model.pkl")

#print the result
print(a_result.predict(test.reshape(1,-1)))


"""
#if the chosen model is "RandomForest",save the prediction
if (model=='Randomforest'): 
    clf=RandomForestClassifier(int(n_estimators))
    clf.fit(stat,y)
    prediction = clf.predict(test.reshape(1,-1))
"""
