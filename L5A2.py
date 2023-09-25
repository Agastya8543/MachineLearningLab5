import numpy as np
import matplotlib.pyplot as plt

# Define the AND gate training data
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
Y = np.array([0, 0, 0, 1])

# Initialize weights and learning rate
W = np.array([10, 0.2])  # Match the input dimensions
learning_rate = 0.05

# Training loop
epochs = 1000
error_history = []

# Define different activation functions
def step_function(x):
    return 1 if x >= 0 else 0

def bipolar_step_function(x):
    if x>0:
        return 1
    elif x==0:
        return 0
    else :
        return -1

def sigmoid_function(x):
    return 1 / (1 + np.exp(-x))

def relu_function(x):
    return x if x>0 else 0

# Create a dictionary to store activation functions
activation_functions = {
    'Step': step_function,
    'Bipolar Step': bipolar_step_function,
    'Sigmoid': sigmoid_function,
    'ReLU': relu_function
}

# Training loop for different activation functions
for activation_name, activation_func in activation_functions.items():
    W = np.array([10, 0.2])  # Reset weights for each activation function
    error_history = []
    
    for epoch in range(epochs):
        total_error = 0
        for i in range(len(X)):
            # Calculate the weighted sum
            weighted_sum = np.dot(X[i], W)
            # Apply the activation function
            output = activation_func(weighted_sum)
            # Calculate the error
            error = Y[i] - output
            total_error += error**2
            # Update weights
            W += learning_rate * error * X[i]
        error_history.append(total_error)
        # Check for convergence
        if total_error <= 0.002:
            break
    
    # Plot the error convergence for each activation function
    plt.plot(range(len(error_history)), error_history, label=activation_name)

# Add labels and legend to the plot
plt.xlabel('Epochs')
plt.ylabel('Error')
plt.title('Error Convergence for Different Activation Functions')
plt.legend()
plt.show()
