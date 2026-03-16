from data.dataset import Dataset

import numpy as np


def train_test_split(data, test_size=0.2, shuffle=False):
    N = len(data)
    
    test_length = int(N * test_size)
    train_length = N - test_length
    
    indices = np.arange(N)
    if shuffle:
        np.random.shuffle(indices)

    train_idx = indices[:train_length]
    test_idx = indices[train_length:]

    train_data = data[train_idx]
    test_data = data[test_idx]

    return Dataset(*train_data), Dataset(*test_data)