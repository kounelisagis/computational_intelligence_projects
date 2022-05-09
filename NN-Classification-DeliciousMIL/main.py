from reader import *
from preprocessor import preprocessor
import neural_net
import embeddings
from sklearn.metrics.pairwise import cosine_similarity


vcbs = read_vocabs()
vcbs_size = len(vcbs)

vectors_option = 'standardization'  # 'normalization' 'centering'

train_text = read_data(filename='data/train-data.dat')
train_vectors = preprocessor(vcbs_size, train_text, option=vectors_option) # X
train_labels = read_labels(filename='data/train-label.dat') # y

test_text = read_data(filename='data/test-data.dat')
test_vectors = preprocessor(vcbs_size, test_text, option=vectors_option) # X
test_labels = read_labels(filename='data/test-label.dat') # y

# OPTION1
# neural_net.train_nn(train_vectors, train_labels, test_vectors, test_labels, vcbs_size)

# OPTION2
# word_to_embedding = embeddings.extract_embeddings(train_text, train_labels, test_text, test_labels, vcbs_size, vcbs)
# notebook = word_to_embedding['notebook']
# textbook = word_to_embedding['textbook']
# dog = word_to_embedding['dog']
# print(cosine_similarity([notebook], [textbook])[0][0], cosine_similarity([notebook], [dog])[0][0])

# OPTION3
embeddings.train_nn(train_text, train_labels, test_text, test_labels, vcbs_size, 'lstm')
