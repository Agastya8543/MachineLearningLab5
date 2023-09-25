import numpy as np
import matplotlib.pyplot as plt

# Initialize weights and bias
W = np.array([10.0, 0.2, -0.75])
learning_rate = 0.05

# Define the step activation function
def step_function(x):
    return 1 if x >= 0 else 0

# Define AND gate input-output pairs
inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
outputs = np.array([0, 0, 0, 1])

# Lists to store error and epochs
error_history = []
epoch_history = []

# Maximum number of iterations
max_epochs = 1000

# Training loop
for epoch in range(max_epochs):
    total_error = 0
    for i in range(len(inputs)):
        input_data = np.insert(inputs[i], 0, 1)  # Add bias term
        net = np.dot(input_data, W)
        output = step_function(net)
        error = outputs[i] - output
        total_error += error**2
        W += learning_rate * error * input_data
    error_history.append(total_error)
    epoch_history.append(epoch)
    
    # Check for convergence
    if total_error <= 0.002:
        print(f"Converged after {epoch} epochs.")
        break

# Plot epochs against error values
plt.plot(epoch_history, error_history)
plt.xlabel('Epochs')
plt.ylabel('Error')
plt.title('Error vs. Epochs')
plt.show()
