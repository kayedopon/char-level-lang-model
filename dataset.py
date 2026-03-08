BLOCK_SIZE = 8

def load_dataset(path="data/sentences.txt"):
    data = []
    to_check = "â€™"
    with open(path, "r") as f:
        i = 0
        for line in f:
            if not to_check in line:
                data.append(line.strip('\n'))
    return data

def get_char_vocab(data):
    chars = set()
    for line in data:
        for char in line:
            chars.add(char)
    chars.add("^") # start token
    chars.add("*") # end token

    return chars

def encode_chars(chars):
    stoi = {c:i for i, c in enumerate(chars)}
    itos = {i:c for c, i in stoi.items()}
    return stoi, itos

def create_samples(data, stoi):
    X, y = [], []
    start_token = stoi["^"]
    for line in data:
        context = [start_token] * BLOCK_SIZE
        for char in line + "*":
            target = stoi[char]
            X.append(context)
            y.append([char])

            context = context[1:] + [char]
    return X, y