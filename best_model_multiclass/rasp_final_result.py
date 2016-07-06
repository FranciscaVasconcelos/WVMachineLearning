from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import numpy as np
from scipy import stats
from sklearn import svm
from sklearn.externals import joblib
from sklearn.cross_validation import train_test_split
import json

#MODULO 1
#LABELS SEPARATED FROM THE MATRIX

#loading data
tmpdata = np.genfromtxt('rasp_data.csv', delimiter=',')
X = np.nan_to_num(tmpdata)


# Creation of labels
tmpdata = np.genfromtxt('rasp_label.csv', delimiter=',')
y = np.nan_to_num(tmpdata)

with open('parameters.json') as data_file:
    dic = json.load(data_file)

model = dic['model']

bin_num = dic['num_bins']
parameter = dic['parameter']

#MODULO2
#PROCESS THE DATA(BINNING)

#binning model matrix,binned matrix==stat 
stat, bin_edges, binnum = stats.binned_statistic(range(X.shape[1]), X, 'median', bins=int(bin_num))
print(stat.shape)


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.33,random_state=42)
#MODULO3
#APPLY THE MODEL AND PRINT THE RESULT

if model == 'svm.LinearSVC()':
    clf=svm.LinearSVC(C=parameter)
if model == 'RandomForestClassifier()': 
    clf=RandomForestClassifier(n_estimators=parameters,n_jobs=-1)
if model == 'LinearDiscriminantAnlysis()': 
    clf=LinearDiscriminantAnalysis()



clf.fit(X_train,y_train)

joblib.dump(clf ,"model.pkl")

a_result=joblib.load("model.pkl")

for i in X_test:
    print(a_result.predict(i.reshape(1,-1)),y_test)

