import numpy as np
from matplotlib import pyplot as plt
from sklearn.manifold import MDS
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn import mixture
import csv



tmpdata = np.genfromtxt('np_specg.csv', delimiter=',')
data = np.nan_to_num(tmpdata)

#pca = PCA(n_components=50)
#pca.fit(data) 


model = MDS(max_iter=200)
np.set_printoptions(suppress=True)
output = model.fit_transform(data)

g = mixture.GMM(n_components=2)
g.fit(output)

output_black = output[0:26]
output_white = output[27:52]
print(output_black)


plt.scatter(output_black[:, 0], output_black[:,1], color='b')
plt.scatter(output_white[:, 0], output_white[:, 1],color='r')
plt.show()

