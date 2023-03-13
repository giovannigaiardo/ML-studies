import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_blobs

from yellowbrick.cluster import KElbowVisualizer
# url = 'https://archive.ics.uci.edu/ml/datasets/seeds#'
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00396/Sales_Transactions_Dataset_Weekly.csv'
df = pd.read_csv(url)

from sklearn.preprocessing import StandardScaler

X = df.iloc[:, 55:107]  # drop the first column

print(X)
scaler = StandardScaler().fit(X)     # fit the scaler to the data
X_scaled = scaler.transform(X)       # apply the scaler to the data

from sklearn.cluster import KMeans

# Generate synthetic dataset with 8 random clusters
X, y = make_blobs(n_samples=811, n_features=52, centers=10, random_state=0)

# Instantiate the clustering model and visualizer
model = KMeans()
visualizer = KElbowVisualizer(model, k=(2,11))

visualizer.fit(X)        # Fit the data to the visualizer
visualizer.show()        # Finalize and render the figure

# kmeans = KMeans(n_clusters=3, random_state=0)  # create a k-means object with 5 clusters
# kmeans.fit(X_scaled)                          # fit the k-means object to the data

# labels = kmeans.labels_         # get the cluster labels for each data point
# centroids = kmeans.cluster_centers_  # get the centroids of each cluster

# # create a scatter plot of the data points
# plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=labels, cmap='viridis')

# # add markers at the positions of the cluster centroids
# plt.scatter(centroids[:, 0], centroids[:, 1], marker='.', s=300, c='red')

# plt.xlabel('Scaled feature 1')
# plt.ylabel('Scaled feature 2')
# plt.title('K-means clustering (k=3)')

# plt.show()

