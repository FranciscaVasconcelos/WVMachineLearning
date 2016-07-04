# Does dimensionality reduction, visualization of data, and classification
# using MDS, GMM, and LDA.

# Import all libraries
import numpy as np
from matplotlib import pyplot as plt
from sklearn.manifold import MDS
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn import mixture
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

# LDA model
lda = LinearDiscriminantAnalysis()
lda.fit(output, labels)
print(lda.predict([[-0.8, -1]]))

# Plotting LDA contour
nx, ny = 200, 100
x_min, x_max = np.amin(output[:,0]), np.amax(output[:,0])
y_min, y_max = np.amin(output[:,1]), np.amax(output[:,1])
xx, yy = np.meshgrid(np.linspace(x_min, x_max, nx),np.linspace(y_min, y_max, ny))
Z = lda.predict_proba(np.c_[xx.ravel(), yy.ravel()])
Z = Z[:, 1].reshape(xx.shape)
plt.contour(xx, yy, Z, [0.5], linewidths=5, colors = 'k', linestyles = 'dashed')

# Plotting LDA means
plt.plot(lda.means_[0][0], lda.means_[0][1],'o', color='black', markersize=10)
plt.plot(lda.means_[1][0], lda.means_[1][1],'o', color='black', markersize=10)
plt.title('LDA with MDS and Gaussian Mixture')

# Plot red and green data
output_red = output[0:26]
output_green = output[27:52]
plt.scatter(output_red[:, 0], output_red[:,1], color='r')
plt.scatter(output_green[:, 0], output_green[:, 1],color='g')
plt.show()

