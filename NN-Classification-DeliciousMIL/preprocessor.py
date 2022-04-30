import numpy as np

def preprocessor(vocab_size, txt, option=None):

    def create_bow_vectors():
        bow_vectors = np.zeros((len(txt), vocab_size))
        for i, paragraph in enumerate(txt):
            for sentence in paragraph:
                bow_vectors[i][sentence] += 1
        
        return bow_vectors

    def normalization(vecs):
        for vec in vecs:
            vec /= np.max(vec)

    def centering(vecs):
        for vec in vecs:
            vec -= np.mean(vec)

    def standardization(vecs):
        centering(vecs)
        normalization(vecs)


    vectors = create_bow_vectors()

    if option == 'normalization':
        normalization(vectors)
    elif option == 'centering':
        centering(vectors)
    elif option == 'standardization':
        standardization(vectors)

    return vectors
