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

learning_rate = 0.0001  # Adjusted learning rate
iterations = 10
m = len(x_train)
cost_history = []

# Perform gradient descent with NumPy optimization
for k in range(iterations):
    # Compute predictions
    y_pred = np.dot(x_train, w) + b
    
    # Compute errors
    errors = y_pred - y_train
    
    # Compute cost
    cost = (1/(2*m)) * np.sum(errors**2)
    cost_history.append(cost)
    
    # Compute gradients using NumPy (Optimized)
    w_grad = (1/m) * np.dot(x_train.T, errors)  # Optimized weight update
    b_grad = (1/m) * np.sum(errors)  # Bias gradient
    
    # Update parameters
    w = w - learning_rate * w_grad
    b = b - learning_rate * b_grad

# Plot cost function
plt.plot(range(1, iterations+1), cost_history, marker='o', linestyle='-')
plt.xlabel("Iteration")
plt.ylabel("Cost")
plt.title("Cost Function Convergence")
plt.show()

# Print final results
print(f"Final Weights: {w}")
print(f"Final Bias: {b}")
