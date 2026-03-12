from .base import Module


class Sequential(Module):
    def __init__(self, *layers):
        super().__init__()
        self.layers = []

        for i, layer in enumerate(layers):
            self.__setattr__(str(i), layer)
            self.layers.append(layer)

    def forward(self, x):
        for l in self.layers:
            x = l.forward(x)
        return x
    
    def backward(self, dout):
        for l in reversed(self.layers):
            dout = l.backward(dout)
        return dout
    
    def chidren(self):
        return self.layers