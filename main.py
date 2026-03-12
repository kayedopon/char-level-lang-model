from utils.data import load_dataset, get_char_vocab, encode_chars, create_samples, train_test_split, BLOCK_SIZE
from utils.dataset import Dataset
from utils.dataloader import DataLoader
from utils.save_load import *

from nn.loss import MulticlassCrossEntropy
from nn.batchnorm import BatchNorm
from nn.loss import MulticlassCrossEntropy
from nn.optim import Adam

from mlp_lm import MLP_LM

from train import train


import numpy as np
import matplotlib.pyplot as plt


def main():
    data = load_dataset()
    length = int(len(data) * 0.1)
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

    res = train(model,train_loader, test_loader, loss_fn, optim, 15)

    fig, axis = plt.subplots(1, 2)
    axis[0].plot(res["train_acc"], label="train")
    axis[0].plot(res["test_acc"], label="test")
    axis[0].set_title("Train vs test acc")
    axis[0].legend()

    axis[1].plot(res["train_loss"], label="train")
    axis[1].plot(res["test_loss"], label="test")
    axis[1].set_title("Train vs test loss")
    axis[1].legend()

    plt.show()


    save_model(model)



if __name__ == "__main__":
    main()