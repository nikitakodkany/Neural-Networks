import numpy as np

def relu(z):
     return np.maximum(0, z)

def relu_prime(z):
    z[z < 0] = 0
    z[z > 0] = 1
    return z

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_prime(z):
  return sigmoid(z) * (1-sigmoid(z))

def softmax(z):
	exp_z = np.exp(z - np.max(z, axis=0))
	return exp_z / exp_z.sum(axis=0, keepdims=True)
