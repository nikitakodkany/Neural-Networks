import numpy as np

def relu(Z):
     return np.maximum(0, Z)

def relu_prime(Z):
    Z[Z < 0] = 0
    Z[Z > 0] = 1
    return Z

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_prime(z):
  return sigmoid(z) * (1-sigmoid(z))
