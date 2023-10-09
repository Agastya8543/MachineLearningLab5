import numpy as np
import matplotlib.pyplot as plt

# Define the XOR gate training data
# Input data (X) and corresponding target labels (Y)
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
Y = np.array([0, 1, 1, 0])

# Initialize weights and bias
weights = np.array([10, 0.2, -0.75])  # Initial weights, including bias
learning_rate = 0.05
epochs = 1000  # Maximum number of epochs
errors = []  # To store the sum-square-error for each epoch

# Step activation function
def step(x):
    return 1 if x > 0 else 0

# Calculate the predicted output
def predict(inputs):
    return step(np.dot(inputs, weights[1:]) + weights[0])

# Update the weights
def update_weights(weights, inputs, targets, learning_rate):
    error = targets - predict(inputs)
    weights[1:] += learning_rate * error * inputs
    weights[0] += learning_rate * error

# Train the perceptron
for epoch in range(epochs):
    error_sum = 0
    for i in range(len(X)):
        error_sum += (predict(X[i]) - Y[i]) ** 2
        update_weights(weights, X[i], Y[i], learning_rate)

    errors.append(error_sum)

    # Check for convergence
    if error_sum == 0:
        print(f"Converged in {epoch + 1} epochs.")
        break

# Print the final weights
print("Final Weights:", weights)

# Plotting the error values
plt.plot(range(1, len(errors) + 1), errors)
plt.xlabel('Epochs')
plt.ylabel('Sum-Square Error')
plt.title('Error Convergence for XOR')
plt.show()


