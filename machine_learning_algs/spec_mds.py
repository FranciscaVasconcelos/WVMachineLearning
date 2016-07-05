import numpy as np
from matplotlib import pyplot as plt
from sklearn.manifold import MDS
from sklearn.decomposition import PCA
import csv

# Get data from csv
tmpdata = np.genfromtxt('np_specg.csv', delimiter=',')
data = np.nan_to_num(tmpdata)

# Apply Multidimensional Scaling
model = MDS(max_iter=200)
np.set_printoptions(suppress=True)
output = model.fit_transform(data)

# Split data into true red and green classes
output_red = output[0:26]
output_green = output[27:52]

# Plot transformed data in scatterplot
plt.scatter(output_red[:, 0], output_red[:,1], color='r')
plt.scatter(output_green[:, 0], output_green[:, 1],color='g')
plt.title('Multidimensional Scaling (MDS)')
plt.show()

