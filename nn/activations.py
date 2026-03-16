import numpy as np


class Tanh():
    """
    The implementation of Tanh activation function.

    Computes:
        `(e^x - e^-x)/(e^x + e^-x)`
    """
    def __init__(self):
        self.out = None
        self.x = None
    
    def forward(self, x):
        self.x = x
        self.out = (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))
        return self.out
    
    def backward(self, dout):
        # the derivative used here was manually calculated and is the same as the known 1 - tanh(x)**2
        return dout * ((4 * np.exp(2 * self.x)) / (np.exp(2 * self.x) + 1) ** 2)
    

def softmax(logits):
    """
    The implementation of softmax activation function.

    Computes:
        e^x / sum(e^x)
    """
    logits = logits - np.max(logits, axis=1, keepdims=True)
    exp = np.exp(logits)
    return exp / np.sum(exp, axis=1, keepdims=True)