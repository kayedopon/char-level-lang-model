from .dataset import Dataset

import numpy as np
import pandas as pd
import random

BLOCK_SIZE = 12
START_TOKEN = "^"
END_TOKEN = "`"
    
# def load_dataset(path="data/datasets/eng_sentences.tsv"):
#     data = []
#     to_check = "â€™"
#     with open(path, "r") as f:
#         i = 0
#         for line in f:
#             if not to_check in line:
#                 data.append(line.strip('\n'))
#     return data

def load_dataset(path="data/datasets/eng_sentences.tsv"):
    data = []
    sentences = pd.read_csv(path, sep="\t").iloc[:, 2]
    for line in sentences:
        data.append(line)
    return data

def get_char_vocab(data):
    chars = set()
    for line in data:
        for char in line:
            chars.add(char)
    chars.add(START_TOKEN) 
    chars.add(END_TOKEN) 

    return sorted(chars) 

def encode_chars(chars):
    stoi = {c:i for i, c in enumerate(chars)}
    itos = {i:c for c, i in stoi.items()}
    return stoi, itos

def create_samples(data, stoi):
    X, y = [], []
    start_token = stoi[START_TOKEN]
    for line in data:
        context = [start_token] * BLOCK_SIZE
        for char in line + END_TOKEN:
            target = stoi[char]
            X.append(context)
            y.append([target])

            context = context[1:] + [target]
    return X, y