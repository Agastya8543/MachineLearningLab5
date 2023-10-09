#8
import numpy as np
import matplotlib.pyplot as plt

# Define the XOR gate training data
# Input data (X) and corresponding target labels (Y)
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
Y = np.array([0, 1, 1, 0])

# Initialize weights and bias
W = np.array([10, 0.2, -0.75])  # Initial weights, including bias
learning_rate = 0.05
epochs = 1000  # Maximum number of epochs
errors = []  # To store the sum-square-error for each epoch

# Step activation function
def step(x):
    return 1 if x > 0 else 0

# Training the perceptron
for epoch in range(epochs):
    error_sum = 0
    for i in range(len(X)):
        # Calculate the predicted output (Y_pred) using the current weights
        Y_pred = step(np.dot(X[i], W[1:]) + W[0])  # Adding bias term
        
        # Calculate the error
        error = Y[i] - Y_pred
        error_sum += error ** 2
        
        # Update weights
        W[1:] += learning_rate * error * X[i]
        W[0] += learning_rate * error  # Update bias
    
    errors.append(error_sum)

    # Check for convergence
    if error_sum == 0:
        print(f"Converged in {epoch + 1} epochs.")
        break

# Plotting the error values
plt.plot(range(1, len(errors) + 1), errors)
plt.xlabel('Epochs')
plt.ylabel('Sum-Square Error')
plt.title('Error Convergence for XOR')
plt.show()

# Print the final weights
print("Final Weights:", W)
