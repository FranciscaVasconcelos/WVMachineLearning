import numpy as np
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import matthews_corrcoef
import csv
from sklearn import svm

tmpdata = np.genfromtxt('np_specg.csv', delimiter=',')
data = np.nan_to_num(tmpdata)

pca = PCA(n_components=2)
comps = pca.fit_transform(data) 

# Creation of labels
labels = []
for i in range(0,27):
    labels.append(1)
for i in range(27,53):
    labels.append(2)

s = svm.SVC(kernel='linear')
s.fit(comps, labels)
y_pred = s.predict(comps)
print(labels)
print(y_pred)
mcc = matthews_corrcoef(labels,y_pred)
print("MCC="+str(mcc))


h = .02  # step size in the mesh
x_min, x_max = comps[:, 0].min() - 1, comps[:, 0].max() + 1
y_min, y_max = comps[:, 1].min() - 1, comps[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),np.arange(y_min, y_max, h))
Z = s.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape) 

plt.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.8)


# Plot red and green data
output_red = comps[0:26]
output_green = comps[27:52]
plt.scatter(output_red[:, 0], output_red[:,1], color='r')
plt.scatter(output_green[:, 0], output_green[:, 1],color='g')
plt.title('PCA with SVM Classification (Linear Kernel)')
plt.show()


