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

def train_model(data, stoi, VOCAB_SIZE, hidden, emb):
    X, y = create_samples(data, stoi)

    data = Dataset(X, y)
    train_data, test_data = train_test_split(data, test_size=0.15, shuffle=True)

    train_loader = DataLoader(train_data, shuffle=True)
    test_loader = DataLoader(test_data)

    model = MLP_LM(BLOCK_SIZE, hidden, VOCAB_SIZE, emb)
    loss_fn = MulticlassCrossEntropy()
    optim = Adam(model.parameters())

    epochs = 10

    res = train(model,train_loader, test_loader, loss_fn, optim, epochs)

    plot_results(res)

    save_model(model, path="models/p.npz")

def main():
    np.random.seed(42)
    data = load_dataset()
    length = int(len(data) * 1)
    data = data[:length]

    chars = get_char_vocab(data)
    VOCAB_SIZE = len(chars)

    stoi, itos = encode_chars(chars)

    hidden = 600
    emb = 100
    train_model(data, stoi, VOCAB_SIZE, hidden, emb)
    model = MLP_LM(BLOCK_SIZE, 600, VOCAB_SIZE, 60)
    load_model(model, path="models/p.npz")
    generate(model, stoi, itos, 20, greedy=False)

if __name__ == "__main__":
    main()