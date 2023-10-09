import numpy as np
import matplotlib.pyplot as plt

# Define the training data for the XOR gate
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
Y = np.array([0, 1, 1, 0])

# Initialize weights and bias
W = np.array([10, 0.2, -0.75])  # W0, W1, W2
learning_rate = 0.05
epochs = 1000

# Initialize lists to store errors and epoch numbers for each activation function
activation_functions = [("Bi-Polar Step", np.sign), ("Sigmoid", lambda x: 1 / (1 + np.exp(-x))), ("ReLU", lambda x: max(0, x))]
errors = {activation: [] for activation, _ in activation_functions}
epoch_numbers = {activation: [] for activation, _ in activation_functions}

# Training the perceptron with different activation functions
for activation, activation_function in activation_functions:
    W = np.array([10, 0.2, -0.75])  # Reset weights
    for epoch in range(epochs):
        error = 0
        for i in range(len(X)):
            # Calculate the weighted sum
            weighted_sum = np.dot(X[i], W[1:]) + W[0]

            # Apply the activation function
            prediction = activation_function(weighted_sum)

            # Calculate the error
            error = error + (Y[i] - prediction) ** 2

            # Update weights
            if activation == "Sigmoid":
                delta = learning_rate * (Y[i] - prediction) * prediction * (1 - prediction)
            elif activation == "ReLU":
                delta = learning_rate * (Y[i] - prediction) if weighted_sum >= 0 else 0
            else:
                delta = learning_rate * (Y[i] - prediction)
            W[1:] = W[1:] + delta * X[i]
            W[0] = W[0] + delta

        # Append error and epoch number for plotting
        errors[activation].append(error)
        epoch_numbers[activation].append(epoch)

        # Check for convergence
        if error <= 0.002:
            print(f"Converged with {activation} after {epoch + 1} epochs")
            break

# Plotting error vs. epoch for all activation functions
plt.figure(figsize=(15, 5))
for i, (activation, _) in enumerate(activation_functions):
    plt.subplot(1, 3, i + 1)
    plt.plot(epoch_numbers[activation], errors[activation])
    plt.xlabel('Epochs')
    plt.ylabel('Error')
    plt.title(f'Error vs. Epochs ({activation})')

plt.tight_layout()
plt.show()
