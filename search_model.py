from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from sklearn import cross_validation as cv
from sklearn import svm
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

#values=[100,500,1000,0,10**-3,10**-2,10**-1,1,10,100,1000]
num_bin=0
max_mcc=0
model=' '
parameter=0
 
def model_run(predictions,y,model_name,val_num,bin_num,nb,maxm,model):
    mcc=metrics.matthews_corrcoef(y, predictions)
    med=np.mean(mcc)
    if med>maxm:
        maxm=med
        parameter=val_num
        model=model_name
        nb=bin_number
    print(model_name,val_num, med)
    return nb, maxm, model,i    

          
for bin_number in range(100,200,5):
    print(bin_number)
    #result of the binning, stat 
    stat, bin_edges, binnum = stats.binned_statistic(range(X.shape[1]), X, 'median', bins=bin_number) 

    ml_models = {'Rand_f': [100,500,1000],'Linear_d': [0],'Support_v':[10**-3,10**-2,10**-1,1,10,100,1000]}

    for key, value in ml_models.items():
        if key == 'Rand_f':
            for i in value:
                predictions= cv.cross_val_predict(RandomForestClassifier(n_estimators=i), stat, y, cv=4)            
                num_bin, max_mcc, model, parameter= model_run(predictions,y,'RandomForest()',i,bin_number,num_bin,max_mcc,model)
        if key == 'Linear_d':
            for i in value:
                predictions= cv.cross_val_predict(LinearDiscriminantAnalysis(), stat, y, cv=4)            
                num_bin, max_mcc, model, parameter= model_run(predictions,y,'LinearDiscriminantAnalysis()',i,bin_number,num_bin,max_mcc,model)
        if key == 'Support_v':
            for i in value:
                predictions= cv.cross_val_predict(svm.LinearSVC(C=i), stat, y, cv=4)            
                num_bin, max_mcc, model, parameter= model_run(predictions,y,'svm.LinearSVC()',i,bin_number,num_bin,max_mcc,model)     
              

print(max_mcc,model,num_bin)

output = {'model': model, 'parameter': parameter, 'num_bins': num_bin}

with open('parameters.json', 'w') as f:
     json.dump(output, f)
#print json.dumps(parameter,model,num_bin)
             
