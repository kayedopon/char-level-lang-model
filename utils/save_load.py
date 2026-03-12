import re
import numpy as np

def is_float_regex(value):
    pattern = r'^-?\d+(\.\d+)?$'
    return bool(re.match(pattern, value))

def save_model(model, path="models/params.txt"):
    with open(path, "w") as f:
        for name, param in model.named_parameters():
            f.write(name + '\n')
            arr = param.value
            for row in arr:
                if arr.ndim == 1:
                    f.write(str(row) + "\n")
                else:
                    f.write(" ".join(map(str, row)) + '\n')
            
def load_model(model, path="models/params.txt"):
    with open(path, "r") as f:
        buffer = []
        label = None
        params = {}

        for line in f:
            line = line.strip('\n').split(' ')
            if line == "":
                continue

            if not is_float_regex(line[0]):
                if label is not None:
                    params[label] = np.array(buffer)
                label = line[0]
                buffer = []
            else:
                buffer.append(list(map(np.float64, line)))
        
        if label is not None:
            params[label] = np.array(buffer)

        for name, param in model.named_parameters():
            param.value = params[name]