import re
import numpy as np

def read_vocabs(filename='data/vocabs.txt'):
    vocabs = {}
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            vocab_name, vocab_index = line.split(', ')
            vocabs[int(vocab_index)] = vocab_name
    return vocabs


def read_data(filename='data/train-data.dat'):
    pattern = re.compile(r"\<(\d+)\>")
    text = []
    with open(filename, 'r') as f:
        for line in f:
            paragraph = line.split()
            indexes = [i for i, item in enumerate(paragraph) if re.search(pattern, item)]
            text.append([paragraph[s+1:e] for s, e in zip(indexes[1:], indexes[2:]+[None])])

    # convert strings to ints inline
    text = [[[int(word) for word in sentence] for sentence in paragraph] for paragraph in text]
    return text


def read_labels_text(filename='data/labels.txt'):
    labels = []
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            label_name, label_index = line.split(', ')
            labels.append( (label_name, label_index, ) )
    return labels


def read_labels(filename='data/train-label.dat'):
    labels = []
    with open(filename, 'r') as f:
        for line in f:
            line_labels = line.strip().split()
            labels.append([int(label) for label in line_labels])
    return np.array(labels)

