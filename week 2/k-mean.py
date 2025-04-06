from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt 
import mglearn

#generate synthetic two-dim data
X, y = make_blobs(random_state=1)

fix, axes = plt.subplots(1, 2, figsize=(10,5))

#build the clustering model
#Kmeans = KMeans(n_clusters=2)
#Kmeans.fit(X)
#assignments = Kmeans.labels_
#mglearn.discrete_scatter(X[:,0], X[:,1], assignments, ax=axes[0])


Kmeans = KMeans(n_clusters=5)
Kmeans.fit(X)
#calculate inertia
inertia = Kmeans.inertia_
print("Inertia:", inertia)

# calculate selhouette score
silhouette = silhouette_score(X, Kmeans.labels_)
print("Silhouette score: ", silhouette)

assignments = Kmeans.labels_
mglearn.discrete_scatter(X[:,0], X[:,1], assignments, ax=axes[1])

#plt.show()
