import numpy as np

from data.dataloader import DataLoader
from nn.base import Module
from nn.loss import MulticlassCrossEntropy
from nn.optim import Adam

def train_step(model: Module, train_dataloader: DataLoader, loss_fn: MulticlassCrossEntropy, optim: Adam):
    """
    Function the implements train step of model training cycle.

    Arguments:
        model (nn.base.Module): model to be trained.
        train_dataloader (data.dataloader.DataLoader): loader that contains train data.
        loss_fn (nn.loss.MulticlassCrossEntropy): loss function to be minimized.
        optim (nn.optim.Adam): optimizer.
    """
    train_acc, train_loss = 0, 0

    model.train()
    for X, y in train_dataloader:
        y = y.reshape(-1)

        logits = model.forward(X)
        loss = loss_fn.forward(logits, y)
        optim.zero_grad()
        dl_dp = loss_fn.backward()
        model.backward(dl_dp)
        optim.step()

        pred = np.argmax(logits, axis=1)
        train_acc += (pred == y).sum() / len(logits)
        train_loss += loss

    train_acc /= len(train_dataloader)
    train_loss /= len(train_dataloader)

    return train_acc, train_loss

def test_step(model: Module, test_dataloader: DataLoader, loss_fn: MulticlassCrossEntropy,):
    """
    Function the implements eval step of model training cycle.

    Arguments:
        model (nn.base.Module): model to be trained.
        test_dataloader (data.dataloader.DataLoader): loader that contains test data.
        loss_fn (nn.loss.MulticlassCrossEntropy): loss function to be minimized.
    """
    test_acc, test_loss = 0, 0

    model.eval()
    for X, y in test_dataloader:
        y = y.reshape(-1)

        logits = model.forward(X)
        loss = loss_fn.forward(logits, y)

        pred = np.argmax(logits, axis=1)
        test_acc += (pred == y).sum() / len(logits)
        test_loss += loss
    
    test_acc /= len(test_dataloader)
    test_loss /= len(test_dataloader)
    
    return test_acc, test_loss

def train(model: Module, train_loader: DataLoader, test_loader: DataLoader, loss_fn: MulticlassCrossEntropy, optim: Adam, epochs: int):
    """
    Function that starts the model training cycle and returns its result.

    Arguments:
        model (nn.base.Module): model to be trained.
        train_dataloader (data.dataloader.DataLoader): loader that contains train data.
        test_dataloader (data.dataloader.DataLoader): loader that contains test data.
        loss_fn (nn.loss.MulticlassCrossEntropy): loss function to be minimized.
        optim (nn.optim.Adam): optimizer.
        epochs (int): number of epochs.
    """
    res = {
        "train_acc": [],
        "train_loss": [],
        "test_acc": [],
        "test_loss": []
    }

    for epoch in range(epochs):
        if epoch + 1 > 0 and (epoch + 1) % 2 == 0:
            optim.lr *= 0.7
            
        train_acc, train_loss = train_step(model, train_loader, loss_fn, optim)
        test_acc, test_loss = test_step(model, test_loader, loss_fn)
        
        print(f"Epoch: {epoch+1} | train_acc: {train_acc:.2f} | train_loss: {train_loss:.4f} | test_acc: {test_acc:.2f} | test_loss: {test_loss:.4f}")

        res["train_acc"].append(train_acc)
        res["train_loss"].append(train_loss)
        res["test_acc"].append(test_acc)
        res["test_loss"].append(test_loss)
    
    return res

