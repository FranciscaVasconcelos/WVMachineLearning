import numpy as np
from matplotlib import pyplot as plt
from sklearn.manifold import MDS
from sklearn.decomposition import PCA
from scipy import stats
import csv

# Get data from csv
tmpdata = np.genfromtxt('np_specg.csv', delimiter=',')
data = np.nan_to_num(tmpdata)

# Apply bins to data
for bin_number in range(20,155,5):
    stat, bin_edges, binnum = stats.binned_statistic(range(data.shape[1]), data, 'median', bins=bin_number)

# Apply Multidimensional Scaling
model = MDS(max_iter=200)
np.set_printoptions(suppress=True)
output = model.fit_transform(stat)

# Split data into true green and red classes
output_red = output[0:26]
output_green = output[27:52]

# Plot transformed data on scatterplot
plt.scatter(output_red[:, 0], output_red[:,1], color='r')
plt.scatter(output_green[:, 0], output_green[:, 1],color='g')
plt.title('Multidimensional Scaling (MDS) with Binning')
plt.show()

