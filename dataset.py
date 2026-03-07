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
    return chars

def encode_chars(chars):
    stoi = {c:i for i, c in enumerate(chars)}
    itos = {i:c for c, i in stoi.items()}
    return stoi, itos
