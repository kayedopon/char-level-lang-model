import numpy as np


class Adam:
    """
    The implementaion of Adam optimizer with weight decay. Used for parameters' values tuning.

    Parameters:
        params (iterable): model's parameters to optimize.
        lr (float, optional): learning rate (default: 1e-3).
        betas (tuple[float, float], optional): cooficients used for the calculation of running averages (default: (0.9, 0.999).
        eps (float, optional): term used for numerical stability (default: 1e-8).
        weight_decay (int, optional): weight decay (default: 0).
    """
    def __init__(self, params, lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0):
        self.params = list(params)
        self.lr = lr
        self.betas = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.t = 0

        self.m = [np.zeros_like(p.value) for p in self.params]
        self.v = [np.zeros_like(p.value) for p in self.params]

    def step(self):
        self.t += 1

        for i, p in enumerate(self.params):
            g = p.grad

            if self.weight_decay != 0:
                g = g + self.weight_decay * p.value

            self.m[i] = self.betas[0] * self.m[i] + (1 - self.betas[0]) * g
            self.v[i] = self.betas[1] * self.v[i] + (1 - self.betas[1]) * g ** 2

            mh = self.m[i] / (1 - self.betas[0] ** self.t)
            vh = self.v[i] / (1 - self.betas[1] ** self.t)

            p.value = p.value - self.lr * mh / (np.sqrt(vh) + self.eps)

    def zero_grad(self):
        for p in self.params:
            p.zero_grad()