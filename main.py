from dataset import load_dataset, get_char_vocab, encode_chars, create_samples
from nn.embedding import Embedding, Flatten

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

if __name__ == "__main__":
    main()