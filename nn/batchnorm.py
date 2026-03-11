from .base import Module, Parameter

import numpy as np


class BatchNorm(Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super().__init__()
        self.gamma = Parameter(np.ones((1, num_features)))
        self.beta =  Parameter(np.zeros((1, num_features)))

        self.eps = eps
        self.momentum = momentum

        self.running_mean = np.ones((1, num_features))
        self.running_var = np.zeros((1, num_features))

        self.x = None
        self.x_hat = None
        self.mean = None
        self.var = None



    def forward(self, x, training=True):
        self.x = x
        print(x.shape)

        if training:
            self.mean = np.mean(x, axis=0, keepdims=True)
            self.var = np.var(x, axis=0, keepdims=True)

            self.x_hat = (x - self.mean) / np.sqrt(self.var + self.eps)

            out = self.gamma * self.x_hat + self.beta

            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * self.mean
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * self.var
        else:
            self.x_hat = (x - self.running_mean) / np.sqrt(self.running_var + self.eps)
            out = self.gamma * self.x_hat + self.beta

        return out
    
    def backward(self, dout):
        N = dout.shape[0]

        self.gamma.grad += (self.x_hat * dout).sum(axis=0, keepdims=True)
        self.beta.grad += dout.sum(axis=0, keepdims=True)

        dx_hat = self.gamma.value * dout
        std_inv = 1.0 / np.sqrt(self.var + self.eps)
        x_mu = self.x - self.mean

        dvar = np.sum(-x_mu / (2 * (self.var + self.eps) ** (3/2)) * dx_hat, 
                      axis=0, 
                      keepdims=True)
        dmean = np.sum(-std_inv * dx_hat, axis=0, keepdims=True) + \
                dvar * np.sum(-2.0 * x_mu, axis=0, keepdims=True) / N
        
        dx = std_inv * dx_hat + dvar * 2.0 * x_mu / N + dmean / N

        return dx
