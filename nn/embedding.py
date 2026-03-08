from nn.base import Module, Parameter

import numpy as np


class Embedding(Module):
    def __init__(self, num_embeddings, embedding_dim):
        super().__init__()
        self.weights = Parameter(np.random.rand(num_embeddings, embedding_dim))
        self.x = None

    def forward(self, x):
        self.x = x
        return self.weights.value[x]
    
    def backward(self, dout):
        demb = np.zeros_like(self.weights.value)
        for i in range(self.x.shape[0]):
            for j in range(self.x.shape[1]):
                xi = self.x[i, j]
                demb[xi] += dout[i, j]
        return demb