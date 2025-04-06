import numpy as np
import matplotlib.pyplot as plt
import sklearn

np.random.seed(42)#to ensure the random data are the same when ever we run the code 
a = np.random.randn(1)#setting random slope as a start
b = np.random.randn(1)#setting a random intercept as a start
x = np.random.rand(100, 1)#setting the random data

learning_rate = 0.01#setting a learning rate 
num_iters = 1000#nb of times to update a and b

y = (3 + 4 * x) + np.random.randn(100, 1) * 0.5#the true relationship with noise to make it realistic

for i in range(num_iters):#looping 
    prediction = a * x + b #compute the y pred   
    loss = np.mean((prediction-y)**2)#using MSE as a loss function

    grad_a = np.mean(2*(prediction-y)*x)#compute the gradient of a (partial derevative of the loss func w.r.t a)
    grad_b = np.mean(2*(prediction-y))#and for b (partial derivative of the loss func w.r.t b)

    a -= learning_rate*grad_a#update a
    b -= learning_rate*grad_b#update b

    if i % 100 == 0: #print the progress every 100 itr.
        print(f'Iters {i}, Loss: {loss}')

print(f'final param: slop = {a}, intercept = {b}')

#evaluate the performance of the model by testing new values 
x_test = np.random.rand(50, 1)
y_test = (3 + 4 * x_test) + np.random.randn(50, 1) * 0.5

# Compute predictions on testing data
predictions_test = a * x_test + b

# Compute loss on testing data (Mean Squared Error)
loss_test = np.mean((predictions_test - y_test) ** 2)
print(f'Testing Loss: {loss_test}')

plt.figure(figsize=(8, 6))
plt.scatter(x, y, label='Data point')
plt.plot(x, prediction, color='red', label='Regression Line')

plt.xlabel('x')
plt.ylabel('y')
plt.title('Scatter Plot of Generated Data')

plt.legend()
plt.grid(True)
plt.show()
