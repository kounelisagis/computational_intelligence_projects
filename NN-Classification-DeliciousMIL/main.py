from reader import *
from preprocessor import preprocessor
from neural_net import create_nn

vcbs = read_vocabs()
vcbs_size = len(vcbs)


train_text = read_data(filename='data/train-data.dat')
train_vectors = preprocessor(vcbs_size, train_text, option='standardization') # X
train_labels = read_labels(filename='data/train-label.dat') # y

test_text = read_data(filename='data/test-data.dat')
test_vectors = preprocessor(vcbs_size, test_text, option='standardization') # X
test_labels = read_labels(filename='data/test-label.dat') # y

create_nn(train_vectors, train_labels, test_vectors, test_labels, vcbs_size)
