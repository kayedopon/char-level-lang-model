from .data import START_TOKEN, END_TOKEN, BLOCK_SIZE

from nn.activations import softmax
from nn.base import Module

import numpy as np


def generate(model: Module, stoi: dict, itos: dict, num_sen:int=20, path:str="gen_sentences.txt"):
    """
    A function that uses the trained model and generate `num_sen` number of sentences
    and save them inside of specified file.

    Arguments:
        model (nn.base.Module) : trained model used to generate sentences.
        stoi (dict) : a dictionary used to convert strings to indices of chars.
        itos (dict) : a dictionary used to convert indices of chars to string.
        num_sen (int, optional): number of sentences to generate.
        path (str, optional): path to the file where generated sentences will be saved.
    """
    sentences = []

    for i in range(num_sen):
        out = ""
        context = np.array([stoi[START_TOKEN]] * BLOCK_SIZE)
        print(context)
        while True:
            logits = model.forward(np.expand_dims(context, axis=0))
            probs = softmax(logits).squeeze(0)
            ix = multinomial(probs)

            context = np.concatenate([context[1:], np.array([ix])])
            if ix == stoi[END_TOKEN]:
                break
            out += itos[ix]
    sentences.append(out)
    
    with open(path, "w") as f:
        for s in sentences:
            f.write(s)


def multinomial(probs):
    """
    Functions that takes probabilties distribution and returns an index of 
    probable to be next character. Higher probability of character means
    it is more likely to be chosen.

    Arguments:
        probs: Probability distribution of characters.
    """
    r = np.random.uniform()
    cum = 0.0

    for i, p in enumerate(probs):
        cum += p
        if r < cum:
            return i
        
    return len(probs) - 1