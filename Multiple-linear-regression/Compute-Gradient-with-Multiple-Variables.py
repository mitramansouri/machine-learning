import numpy as np
import matplotlib.pyplot as plt

# Training Data (House Size, Bedrooms, Age)
x_train = np.array([[50, 2, 10], 
                    [75, 3, 15], 
                    [100, 4, 20]])
y_train = np.array([100, 150, 200])

# Initialize Parameters
b = 0
w = np.zeros(3)  # Weights for 3 features

learning_rate = 0.001  # Increased learning rate
iterations = 10
m = len(x_train)
n = len(x_train[0])
cost_history = []

# Perform gradient descent with multiple variables
for k in range(iterations):
    # Compute y_pred
    y_pred = np.dot(x_train, w) + b
    
    # Compute Errors
    errors = y_pred - y_train
    
    # Compute MSE
    cost = (1/(2*m)) * np.sum(errors**2)
    cost_history.append(cost)
    
    # Compute W-grad , b-grad
    for j in range(n):
        w_grad = 0
        for i in range(m):
            w_grad+= errors[i]* x_train[i][j]
        w[j] = w[j] - learning_rate * (1/m) * w_grad
    
    b = b - learning_rate * (1/m) * np.sum(errors)
    
# ploting Cost function
plt.plot(range(1, iterations+1), cost_history, marker='o', linestyle='-')
plt.show()

            
    
    

