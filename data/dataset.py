import numpy as np


class Dataset:
    def __init__(self, X, y):
        self.X = np.array(X)
        self.y = np.array(y)

    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, key):
        return self.X[key], self.y[key]