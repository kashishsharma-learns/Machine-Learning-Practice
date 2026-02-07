import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

x,y=make_blobs(n_samples=300,centers=3,cluster_std=1.0,random_state=42)

k=3
kmeans=KMeans(n_clusters=k,random_state=42,n_init=10)
kmeans.fit(x)

centers=kmeans.cluster_centers_
labels=kmeans.labels_

plt.figure(figsize=(8,6))
plt.scatter(x[:,0],x[:,1],c=labels,cmap='viridis',s=30,alpha=0.6)
plt.scatter(centers[:,0],centers[:,1],c='red',s=200,marker='X',label='Centroids')
plt.title("K-means Clustering(k=3)")
plt.legend()

plt.show()
