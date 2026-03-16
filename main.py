from data.data import load_dataset, get_char_vocab, encode_chars, create_samples, BLOCK_SIZE
from data.dataset import Dataset
from data.dataloader import DataLoader
from utils.save_load import save_model, load_model

from inference.generate import generate

from utils.split import train_test_split
from utils.plot import plot_results

from nn.loss import MulticlassCrossEntropy
from nn.optim import Adam

from mlp_lm import MLP_LM

from train import train

import numpy as np
import matplotlib.pyplot as plt


def main():
    np.random.seed(42)
    data = load_dataset()
    length = int(len(data) * 0.03)
    data = data[:length]

    chars = get_char_vocab(data)
    VOCAB_SIZE = len(chars)

    stoi, itos = encode_chars(chars)
    X, y = create_samples(data, stoi)

    data = Dataset(X, y)
    train_data, test_data = train_test_split(data, test_size=0.1, shuffle=True)

    train_loader = DataLoader(train_data, shuffle=True)
    test_loader = DataLoader(test_data)

    model = MLP_LM(BLOCK_SIZE, 256, VOCAB_SIZE, 50)
    loss_fn = MulticlassCrossEntropy()
    optim = Adam(model.parameters())

    res = train(model,train_loader, test_loader, loss_fn, optim, 5)

    plot_results(res)

    save_model(model)

    generate(model, stoi, itos, 20)


if __name__ == "__main__":
    main()