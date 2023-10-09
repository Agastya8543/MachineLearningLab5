import numpy as np
import matplotlib.pyplot as plt

# Step activation function
def step_function(x):
    return 1 if x > 0 else 0

# Perceptron training algorithm
def train_perceptron(X, Y, w0, w1, w2, learning_rate, epochs):
    errors = []

    for epoch in range(epochs):
        total_error = 0

        for i in range(len(X)):
            # Calculate the weighted sum
            weighted_sum = w0 + X[i][0] * w1 + X[i][1] * w2

            # Calculate the predicted output using the step function
            prediction = step_function(weighted_sum)

            # Calculate the error
            error = Y[i] - prediction

            # Update weights and bias
            w0 += learning_rate * error
            w1 += learning_rate * error * X[i][0]
            w2 += learning_rate * error * X[i][1]

            total_error += error ** 2

        errors.append(total_error)

        # Check if the error is zero (converged)
        if total_error == 0:
            break

    return w0, w1, w2, errors

def main():
    # Intitialise weights, learning rate
    w0 = 10
    w1 = 0.2
    w2 = -0.75
    learning_rate = 0.05

    # Define XOR gate input data and expected output
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    Y = np.array([0, 1, 1, 0])

    # Train the perceptron
    epochs = 1000  # You may need to adjust the number of epochs
    final_w0, final_w1, final_w2, errors = train_perceptron(X, Y, w0, w1, w2, learning_rate, epochs)

    # Plot the errors
    plt.plot(range(len(errors)), errors)
    plt.xlabel('Epochs')
    plt.ylabel('Error')
    plt.title('Error vs. Epochs')
    plt.show()

    print(f"Final Weights w0 = {final_w0} , w1 = {final_w1} , w2 = {final_w2} ")

if __name__ == "__main__":
    main()
