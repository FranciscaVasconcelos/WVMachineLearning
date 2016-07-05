from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from sklearn import cross_validation
from sklearn.lda import LDA

#MODULO 1
#LABELS SEPARATED FROM THE MATRIX


#loading data
tempdata = np.genfromtxt("np_specg.csv",  delimiter=',')
X=np.nan_to_num(tempdata)

#load training set label
tempt = np.genfromtxt("labels.csv", delimiter=',')
y=np.nan_to_num(tempt)


#implement the label in the matrix(x_tr)
#x_tr = np.concatenate((x_tr, y_tr.reshape((-1,1))), axis=1)


#do the binning for x_tr

#define window length
ml_models = {'Random Forest 100': [RandomForestClassifier(n_estimators=100)],
             'Random Forest 500': [RandomForestClassifier(n_estimators=500)],
             'Random Forest 1000': [RandomForestClassifier(n_estimators=1000)],
             #'Support Vector Classifier': [SVC(kernel='linear',C=10**i) for i in np.arange(-3,4)],
             #'Linear Discriminant Analysis':[LDA()]
            }
for bin_number in range(20,155,5):
    print(bin_number)

#result of the binning, stat 
    stat, bin_edges, binnum = stats.binned_statistic(range(X.shape[1]), X, 'median', bins=bin_number)
#print the graphs
#for i in range(stat.shape[0]):
#	plt.plot(stat[i])
#really print the graphs	
#plt.show()

    for keys, models in ml_models.items():
        for model in models:
            predictions= cross_validation.cross_val_predict(model, X, y, cv=5)
            mcc=metrics.matthews_corrcoef(y, predictions)
            med=np.mean(mcc)
            print(keys, med)

'''
clf.fit(stat,y_tr)#x_tr
y_pred = clf.predict(stat_ts)#x_ts

mcc=metrics.matthews_corrcoef(y_ts, y_pred)

ml_models = {}
print(mcc)
'''
'''
if keys=='Linear Discriminant Analysis':
            ld=LinearDiscriminantAnalysis()
            predictions=cross_validation.cross_val_predict(ld.fit(X,y).predict(X), X, y, cv=5)
            mcc=metrics.matthews_corrcoef(y, predictions)
            med=np.mean(mcc)
            print(keys, med)
'''


































feat_tr = data_tr[1]
samp_tr= data_tr[1]
#load test set data and labels

data_ts = np.genfromtxt("arcene_valid.data",dtype=bytes, delimiter=' ')
x_ts=data_ts.astype(np.int)    #[1].astype(bytes)
y_ts=np.genfromtxt("arcene_valid.labels",dtype=bytes, delimiter=' ')
y_ts = y_ts.astype(np.int)

 
stat_ts, bin_edges_ts, binnum_ts = stats.binned_statistic(range(x_ts.shape[1]), x_ts, 'median', bins=100)
#print the graphs
for i in range(stat_ts.shape[0]):
	plt.plot(stat_ts[i])
#really print the graphs	
plt.show()



clf=RandomForestClassifier()
clf.fit(stat,y_tr)#x_tr
y_pred = clf.predict(stat_ts)#x_ts

mcc=metrics.matthews_corrcoef(y_ts, y_pred)

ml_models = {}
print(mcc)



