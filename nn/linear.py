import numpy as np

from .base import Parameter, Module


class Linear(Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()

        limit = 1 / np.sqrt(in_dim)
        self.weights = Parameter(np.random.uniform(-limit, limit, (in_dim, out_dim))) # decided to uniformly distribute weights this time
        self.bias = Parameter(np.zeros((1, out_dim)))
        self.x = None

    def forward(self, x):
        self.x = x
        return x @ self.weights.value + self.bias.value
    
    def backward(self, dout):
        self.weights.grad = self.x.T @ dout
        self.bias.grad = np.sum(dout, axis=0, keepdims=True)
        return dout @ self.weights.value.T