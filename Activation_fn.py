import numpy as np
import matplotlib.pyplot as plt

# Define the activation functions
def sigmoid(x):
  return 1 / (1 + np.exp(-x))

def tanh(x):
  return np.sinh(x) / np.cosh(x)

def relu(x):
  return np.maximum(0, x)

def leaky_relu(x):
  return np.maximum(0.01 * x, x)

# Generate data for the plots
x = np.linspace(-10, 10, 100)

# Create the plots
fig, axes = plt.subplots(2, 2, figsize=(10, 10))

# Plot the sigmoid function
axes[0, 0].plot(x, sigmoid(x))
axes[0, 0].set_title('Sigmoid')

# Plot the tanh function
axes[0, 1].plot(x, tanh(x))
axes[0, 1].set_title('Tanh')

# Plot the ReLU function
axes[1, 0].plot(x, relu(x))
axes[1, 0].set_title('ReLU')

# Plot the leaky ReLU function
axes[1, 1].plot(x, leaky_relu(x))
axes[1, 1].set_title('Leaky ReLU')

# Show the plots
plt.show()

# bug fixed in main branch file