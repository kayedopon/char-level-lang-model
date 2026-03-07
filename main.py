from dataset import load_dataset, get_char_vocab, encode_chars


def main():
    data = load_dataset()

    chars = get_char_vocab(data)
    stoi, itos = encode_chars(chars)

if __name__ == "__main__":
    main()