from nn.sequential import Sequential
from nn.embedding import Embedding, Flatten
from nn.linear import Linear
from nn.batchnorm import BatchNorm
from nn.activations import Tanh
from nn.base import Module


class MLP_LM(Module):
    def __init__(self, block_size, hidden_units, vocab_size, emb_dim):
        super().__init__()
        self.net = Sequential(
            Embedding(vocab_size, emb_dim),
            Flatten(),
            Linear(block_size*emb_dim, out_dim=hidden_units),
            BatchNorm(hidden_units),
            Tanh(),
            Linear(hidden_units, vocab_size)
        )

    def forward(self, x):
        return self.net.forward(x)
    
    def backward(self, dout):
        return self.net.backward(dout)