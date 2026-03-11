from utils.data import load_dataset, get_char_vocab, encode_chars, create_samples, train_test_split, BLOCK_SIZE
from utils.dataset import Dataset
from utils.dataloader import DataLoader

from nn.loss import MulticlassCrossEntropy
from nn.batchnorm import BatchNorm
from nn.loss import MulticlassCrossEntropy
from nn.optim import Adam

from mlp_lm import MLP_LM

from train import train_step


import numpy as np


def main():
    data = load_dataset()

    chars = get_char_vocab(data)
    VOCAB_SIZE = len(chars)

    stoi, itos = encode_chars(chars)
    X, y = create_samples(data, stoi)
    data = Dataset(X, y)

    train_data, test_data = train_test_split(data, shuffle=False)

    train_loader = DataLoader(train_data, shuffle=True)
    test_loader = DataLoader(test_data)
    
    model = MLP_LM(8, 160, VOCAB_SIZE, 10)
    
    loss_fn = MulticlassCrossEntropy()
    optim = Adam(model.parameters())

    train_step(model,train_loader, loss_fn, optim)


if __name__ == "__main__":
    main()