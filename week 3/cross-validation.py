from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import numpy as np


# Assuming the linear regression model has been trained and parameters 'a' and 'b' have been obtained

x = np.random.rand(100,1)
y = (3+4*x)+np.random.randn(100,1)*0.5

kf=KFold(n_splits=5)
mse_values = []

for train_index, test_index in kf.split(x):
    x_train, x_test = x[train_index], x[test_index]
    y_train, y_test = y[train_index], y[test_index]

#pred_test = a*x_test + b

#compute avg MSE across all folds
avg_mse = np.mean(mse_values)

print(f'average MSE={avg_mse}')
