from .base import Module

import numpy as np


class MulticlassCrossEntropy(Module):
    """
    Implements Multiclass cross entropy.
    Requires logits as input for prediction since it's internally using softmax.

    Arguments:
        p: logits
        y: targets
    """
    def __init__(self):
        self.p = None
        self.y = None

    def forward(self, p, y):
        self.y = y
        # softmax
        p = p - np.max(p, axis=1, keepdims=True)
        self.p = np.exp(p) / np.sum(np.exp(p), axis=1, keepdims=True)

        # finding log probs of correct classes
        logprobs = np.log(self.p)
        loss = -logprobs[np.arange(len(y)), y]

        return loss.mean()
    
    def backward(self):
        y_onehot = np.zeros_like(self.p)
        y_onehot[np.arange(len(self.y)), self.y] = 1
        dout = self.p - y_onehot
        return dout / len(self.y)

