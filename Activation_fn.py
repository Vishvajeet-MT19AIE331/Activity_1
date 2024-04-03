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

# Get data for the plots
random_values = np.array([-3.5, -1.2, 0, 2.8, -4.1, 1.5, -0.7, 3.2, -2.4, 4.6])

print('Output of tanh function for given values:', tanh(random_values))
print()
print('Output of ReLU function for given values:', relu(random_values))
print()
print('Output of leaky ReLU function for given values:', leaky_relu(random_values))
print()