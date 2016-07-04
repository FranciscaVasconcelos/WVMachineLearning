import numpy as np
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
import csv
from mpl_toolkits.mplot3d import Axes3D



tmpdata = np.genfromtxt('np_specg.csv', delimiter=',')
data = np.nan_to_num(tmpdata)

pca = PCA(n_components=2)
comps = pca.fit_transform(data) 


#plt.plot(pca.explained_variance_, linewidth=2)
#plt.title('Principal Component Analysis (PCA) Feature Assessment')

# Creation of labels
labels = []
for i in range(0,27):
    labels.append(1)
for i in range(27,53):
    labels.append(2)

# LDA model
lda = QuadraticDiscriminantAnalysis()
lda.fit(comps, labels)


# Plotting LDA contour
nx, ny = 200, 100
x_min, x_max = np.amin(comps[:,0]), np.amax(comps[:,0])
y_min, y_max = np.amin(comps[:,1]), np.amax(comps[:,1])
xx, yy = np.meshgrid(np.linspace(x_min, x_max, nx),np.linspace(y_min, y_max, ny))
Z = lda.predict_proba(np.c_[xx.ravel(), yy.ravel()])
Z = Z[:, 1].reshape(xx.shape)
plt.contour(xx, yy, Z, [0.5], linewidths=5, colors = 'k', linestyles = 'dashed')

# Plotting LDA means
plt.plot(lda.means_[0][0], lda.means_[0][1],'o', color='black', markersize=10)
plt.plot(lda.means_[1][0], lda.means_[1][1],'o', color='black', markersize=10)
plt.title('PCA with QDA')

# Plot red and green data
output_red = comps[0:26]
output_green = comps[27:52]
plt.scatter(output_red[:, 0], output_red[:,1], color='r')
plt.scatter(output_green[:, 0], output_green[:, 1],color='g')
plt.show()


