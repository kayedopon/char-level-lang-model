import re
import numpy as np


def save_model(model, path="models/params.npz"):
    params = {}
    for name, param in model.named_parameters():
        params[name] = param.value
    np.savez(path, **params)
            
def load_model(model, path="models/params.npz"):
    params = np.load(path)
    for name, param in model.named_parameters():
        if name not in params:
            raise KeyError(f"Missing parameter: {name}")
        if param.value.shape != params[name].shape:
            raise ValueError(
                f"Shape mismatch for {name}: "
                f"expected {param.value.shape}, got {params[name].shape}"
            )
        param.value[...] = params[name]