import numpy as np

def sigmoid(z):
    return 1/(1+np.exp(-z))

z_linear = np.array([0.5, -1.2, 2.3])

probabilities = sigmoid(z_linear)
print(probabilities)