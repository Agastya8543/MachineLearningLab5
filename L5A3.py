import numpy as np
import matplotlib.pyplot as plt

# Step activation function
def step_activation(x):
    return 1 if x >= 0 else 0

# AND gate training data
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([0, 0, 0, 1])

# Initial weights
initial_weights = np.array([10, 0.2, -0.75])

# List of learning rates to test
learning_rates = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]

# Dictionary to store the number of iterations for each learning rate
iterations_required = {}

for learning_rate in learning_rates:
    # Clone initial weights for each learning rate
    W = np.copy(initial_weights)
    
    # Counter for the number of iterations
    iterations = 0
    
    while True:
        # Calculate the weighted sum for all examples at once
        weighted_sum = np.dot(X, W[1:]) + W[0]
            
        # Apply the step activation function for all examples at once
        output = np.where(weighted_sum >= 0, 1, 0)
            
        # Calculate the error for all examples at once
        delta = y - output
            
        # Update weights
        W[1:] += learning_rate * np.dot(X.T, delta)
        W[0] += learning_rate * np.sum(delta)
            
        # Update error
        error = np.sum(delta ** 2)
        
        # Check for convergence
        if error == 0:
            break
        
        iterations += 1
    
    iterations_required[learning_rate] = iterations
    
# Print the number of iterations for each learning rate
for learning_rate, num_iterations in iterations_required.items():
    print(f"Learning Rate {learning_rate}: {num_iterations} iterations")

# Plot the number of iterations vs. learning rates
plt.plot(learning_rates, [iterations_required[lr] for lr in learning_rates], marker='o')
plt.xlabel('Learning Rate')
plt.ylabel('Number of Iterations for Convergence')
plt.title('Number of Iterations vs. Learning Rate')
plt.grid(True)
plt.show()

