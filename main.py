from dataset import load_dataset, get_char_vocab, encode_chars, create_samples
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

    c = Embedding(len(stoi,), 10)
    x = np.random.randint(0, 72, (2, 8))
    emb = c.forward(x)
    fl = Flatten()
    embcat = fl.forward(emb)
    print(embcat.shape)
    seq = Sequential(
        Embedding(72, 10),
        Flatten(),
        Linear(80, 100),
        Tanh(),
        Linear(80, 1)
    )

if __name__ == "__main__":
    main()