from sklearn.model_selection import learning_curve
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import numpy as np

x = np.random.rand(100,1)
y = (3 + 4*x) + np.random.randn(100,1)*0.5

model = LinearRegression()

model.fit(x, y)

train_sizes, train_scores, test_scores = learning_curve(estimator=model,X=x, y=y, train_sizes=np.linspace(0.1,1.0,10), scoring='neg_mean_squared_error')

train_scores_mean = np.mean(-train_scores, axis=1)
test_scores_mean = np.mean(-test_scores, axis=1)

plt.figure(figsize=(10,6))
plt.plot(train_sizes, train_scores_mean, label='Training error')
plt.plot(train_sizes, test_scores_mean, label='Validation error')
plt.xlabel('Training examples')
plt.ylabel('Negative Mean Squared Error')
plt.title('Learning Curve')
plt.legend()
plt.grid()
plt.show()