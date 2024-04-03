import numpy as np
import matplotlib.pyplot as plt

# Define the activation functions
def sigmoid(x):
  return 1 / (1 + np.exp(-x))

# Generate data for the plots
random_values = np.array([-3.5, -1.2, 0, 2.8, -4.1, 1.5, -0.7, 3.2, -2.4, 4.6])

# print the output of the function
print('Output of Sigmoid function for given values:', sigmoid(random_values))