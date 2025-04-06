
import numpy as np

from sklearn.model_selection import train_test_split

from sklearn.neighbors import KNeighborsClassifier



# Create synthetic dataset

X = np.array([[1], [2], [3], [4], [5]])

y = np.array([0, 1, 0, 1, 0])



# Implement train-test split

X_train, X_test, y_train, y_test = train_test_split(X,y,random_state=0)



# Initialize k-NN classifier

knn = KNeighborsClassifier(n_neighbors=3)



# Train the k-NN classifier using the training data

knn.fit(X_train, y_train)



# Validate the k-NN classifier

accuracy = knn.score(X_test, y_test) 



print(f"Validation Accuracy: {accuracy}")
