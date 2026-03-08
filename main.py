from utils.data import load_dataset, get_char_vocab, encode_chars, create_samples
from utils.dataset import Dataset
from utils.dataloader import DataLoader
from nn.embedding import Embedding, Flatten
from nn.sequential import Sequential
from nn.activations import Tanh
from nn.linear import Linear


import numpy as np


def main():
    data = load_dataset()

    chars = get_char_vocab(data)
    stoi, itos = encode_chars(chars)
    X, y = create_samples(data, stoi)
    data = Dataset(X, y)

    loader = DataLoader(data)
    

if __name__ == "__main__":
    main()