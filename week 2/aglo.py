from sklearn.cluster import AgglomerativeClustering
import matplotlib.pyplot as plt
import mglearn
from sklearn.datasets import make_blobs

#generate random 2D data
X, y = make_blobs(random_state=1)

#perform agglo clustring with 3 clusters
agg = AgglomerativeClustering(n_clusters=3)
assignment = agg.fit_predict(X)

#plot the cluster assignment
mglearn.discrete_scatter(X[:,0], X[:,1], assignment)
plt.xlabel('Feature 0')
plt.ylabel('Feature 1')
plt.title('cluster assignment using agglo clustring with three clusters')
plt.show()