import numpy as np
import matplotlib.pyplot as plt

# Define the AND gate training data
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
Y = np.array([0, 0, 0, 1])

# Initialize weights and learning rate
W = np.array([10, 0.2])  # Update to match the input dimensions
learning_rate = 0.05

# Define the step activation function
def step_function(x):
    return 1 if x >= 0 else 0

# Training loop
epochs = 1000
error_history = []

for epoch in range(epochs):
    total_error = 0
    for i in range(len(X)):
        # Calculate the weighted sum
        weighted_sum = np.dot(X[i], W)
        # Apply the step activation function
        output = step_function(weighted_sum)
        # Calculate the error
        error = Y[i] - output
        total_error += error**2
        # Update weights
        W += learning_rate * error * X[i]
    error_history.append(total_error)
    # Check for convergence
    if total_error <= 0.002:
        break

# Plot the error convergence
plt.plot(range(len(error_history)), error_history)
plt.xlabel('Epochs')
plt.ylabel('Error')
plt.title('Error Convergence')
plt.show()

print(f'Converged after {epoch+1} epochs')
print('Final Weights:', W)
