from .base import Module, Parameter

import numpy as np


class Embedding(Module):
    """
    Embedding layer mapping token indices to dense vectors.
    """
    def __init__(self, num_embeddings, embedding_dim):
        super().__init__()
        self.weights = Parameter(np.random.rand(num_embeddings, embedding_dim) * 0.01)
        self.x = None

    def forward(self, x):
        self.x = x
        return self.weights.value[x]
    
    def backward(self, dout):
        demb = np.zeros_like(self.weights.value)
        np.add.at(demb, self.x, dout)
        self.weights.grad = demb
        return demb
    

class Flatten:
    def __init__(self, start_dim=1, end_dim=-1):
        self.start_dim = start_dim
        self.end_dim = end_dim
        self.orig_shape = None
    
    def forward(self, x):
        self.orig_shape = x.shape
        flatten_size = 1

        if self.end_dim != -1:
            end = self.end_dim + 1
        else:
            end = len(x.shape)

        for dim in x.shape[self.start_dim:end]:
            flatten_size *= dim
        
        shape = (*x.shape[:self.start_dim], flatten_size, *x.shape[end:])
        x = x.reshape(shape)
        return x
    
    def backward(self, dout):
        return dout.reshape(self.orig_shape)
        