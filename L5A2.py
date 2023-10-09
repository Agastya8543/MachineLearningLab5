import numpy as np
import matplotlib.pyplot as plt

# Define the training data for the AND gate
input_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
target_output = np.array([0, 0, 0, 1])

# Initialize weights and hyperparameters
initial_weights = np.array([10, 0.2, -0.75])  # W0, W1, W2
learning_rate = 0.05
epochs = 1000

# Define activation functions
def bipolar_step_activation(x):
    return 1 if x >= 0 else -1

def sigmoid_activation(x):
    return 1 / (1 + np.exp(-x))

def relu_activation(x):
    return max(0, x)

# Training function
def train_perceptron(input_data, target_output, activation_function, initial_weights, learning_rate, epochs):
    weights = initial_weights.copy()
    errors = []
    epoch_numbers = []

    for epoch in range(epochs):
        total_error = 0
        for i in range(len(input_data)):
            # Calculate the weighted sum
            weighted_sum = np.dot(input_data[i], weights[1:]) + weights[0]

            # Apply the activation function
            prediction = activation_function(weighted_sum)

            # Calculate the error
            error = (target_output[i] - prediction) ** 2

            # Update weights
            delta = learning_rate * (target_output[i] - prediction)
            weights[1:] += delta * input_data[i]
            weights[0] += delta

            total_error += error

        # Append error and epoch number for plotting
        errors.append(total_error)
        epoch_numbers.append(epoch)

        # Check for convergence
        if total_error <= 0.002:
            print(f"Converged with {activation_function.__name__} after {epoch + 1} epochs")
            break

    return errors, epoch_numbers

# Train with different activation functions
errors_bipolar_step, epoch_numbers_bipolar_step = train_perceptron(input_data, target_output, bipolar_step_activation, initial_weights, learning_rate, epochs)
errors_sigmoid, epoch_numbers_sigmoid = train_perceptron(input_data, target_output, sigmoid_activation, initial_weights, learning_rate, epochs)
errors_relu, epoch_numbers_relu = train_perceptron(input_data, target_output, relu_activation, initial_weights, learning_rate, epochs)

# Plotting error vs. epoch for all activation functions
plt.figure(figsize=(12, 6))
plt.subplot(131)
plt.plot(epoch_numbers_bipolar_step, errors_bipolar_step)
plt.xlabel('Epochs')
plt.ylabel('Error')
plt.title('Error vs. Epochs (Bi-Polar Step)')

plt.subplot(132)
plt.plot(epoch_numbers_sigmoid, errors_sigmoid)
plt.xlabel('Epochs')
plt.ylabel('Error')
plt.title('Error vs. Epochs (Sigmoid)')

plt.subplot(133)
plt.plot(epoch_numbers_relu, errors_relu)
plt.xlabel('Epochs')
plt.ylabel('Error')
plt.title('Error vs. Epochs (ReLU)')

plt.tight_layout()
plt.show()
