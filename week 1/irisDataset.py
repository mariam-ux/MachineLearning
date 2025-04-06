from sklearn.datasets import load_iris # type: ignore
iris_data = load_iris()
print("keys of iris_dataset: \n{}".format(iris_data.keys()))

print("Traget names: {}".format(iris_data['target_names']))
print("Feature names: {}".format(iris_data['feature_names']))
print("Data: {}".format(iris_data['data']))
print("Data shape: {}".format(iris_data['data'].shape))
print("Data firts 5 columns: {}".format(iris_data['data'][:5]))

#visualization of the data set 
import pandas as pd
import mglearn
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

iris_dataframe = pd.DataFrame(iris_data['data'], columns=iris_data.feature_names)

grr = pd.plotting.scatter_matrix(iris_dataframe, c=iris_data['target'], figsize=(15,15),marker='o', hist_kwds={'bins':20},s=60,alpha=0.8,cmap=mglearn.cm3)

#spliting the model
X_train, X_test, y_train, y_test = train_test_split(iris_data['data'], iris_data['target'], random_state=0)

#creating the classifier
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train,y_train)#here we are training the model

print("Test set score: {:.2f}".format(knn.score(X_test, y_test)))
