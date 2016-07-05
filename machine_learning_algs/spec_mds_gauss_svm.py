# Does dimensionality reduction, visualization of data, and classification
# using MDS, GMM, and SVM.

# Import all libraries
import numpy as np
from matplotlib import pyplot as plt
from sklearn.manifold import MDS
from sklearn.decomposition import PCA
from sklearn import mixture
from sklearn import svm
import csv

# Get data from csv
tmpdata = np.genfromtxt('np_specg.csv', delimiter=',')
data = np.nan_to_num(tmpdata)

# MDS model
model = MDS(max_iter=200)
np.set_printoptions(suppress=True)
output = model.fit_transform(data)

# Gaussian Mixture model
g = mixture.GMM(n_components=2)
g.fit(output)

# Creation of labels
labels = []
for i in range(0,27):
    labels.append(1)
for i in range(27,53):
    labels.append(2)

# Apply support vector machine
s = svm.SVC()
s.fit(output, labels)

# Plot SVM contour
h = .02  # step size in the mesh
x_min, x_max = output[:, 0].min() - 1, output[:, 0].max() + 1
y_min, y_max = output[:, 1].min() - 1, output[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),np.arange(y_min, y_max, h))
Z = s.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.8)

# Plot red and green data
output_red = output[0:26]
output_green = output[27:52]
plt.scatter(output_red[:, 0], output_red[:,1], color='r')
plt.scatter(output_green[:, 0], output_green[:, 1],color='g')
plt.show()

