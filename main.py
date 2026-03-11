from utils.data import load_dataset, get_char_vocab, encode_chars, create_samples
from utils.dataset import Dataset
from utils.dataloader import DataLoader

from nn.loss import MulticlassCrossEntropy
from nn.batchnorm import BatchNorm


import numpy as np


def main():
    data = load_dataset()

    chars = get_char_vocab(data)
    stoi, itos = encode_chars(chars)
    X, y = create_samples(data, stoi)
    data = Dataset(X, y)

    loader = DataLoader(data)
    p = np.array([[0.3432, 0.4654, 0.2123], [1.2342, 0.5342, 0.8324]])
    y = np.array([1, 0])



if __name__ == "__main__":
    main()