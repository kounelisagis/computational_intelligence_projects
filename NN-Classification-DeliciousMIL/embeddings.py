import tensorflow as tf
from keras.layers import Dense, Flatten, LSTM
from tensorflow_addons.optimizers import SGDW
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding
from collections import defaultdict
from sklearn.model_selection import KFold


def get_vectors(train_text, test_text, vcbs_size):

    flat_train_text = [[word for sentence in paragraph for word in sentence] for paragraph in train_text]
    flat_test_text = [[word for sentence in paragraph for word in sentence] for paragraph in test_text]

    input_dim = vcbs_size
    output_dim = 50
    max_length = max([len(paragraph) for paragraph in flat_train_text] + [len(paragraph) for paragraph in flat_test_text])

    train_vectors = pad_sequences(flat_train_text, maxlen=max_length, padding='post')
    test_vectors = pad_sequences(flat_test_text, maxlen=max_length, padding='post')

    return train_vectors, test_vectors, input_dim, output_dim, max_length


def extract_embeddings(train_text, train_labels, test_text, test_labels, vcbs_size, word_to_index):

    train_vectors, _, input_dim, output_dim, max_length = get_vectors(train_text, test_text, vcbs_size)

    model = Sequential()
    model.add(Embedding(input_dim=input_dim, output_dim=output_dim, input_length=max_length, name='embeddings'))
    model.add(Flatten())
    model.add(Dense(20, activation='sigmoid'))

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(train_vectors, train_labels, epochs=20, verbose=1)

    embeddings = model.get_layer('embeddings').get_weights()[0]
    word_to_embedding = {w:embeddings[idx] for w, idx in word_to_index.items()}

    return word_to_embedding


def train_nn(train_text, train_labels, test_text, test_labels, vcbs_size, option='normal'):

    train_vectors, test_vectors, input_dim, output_dim, max_length = get_vectors(train_text, test_text, vcbs_size)

    hidden_layer1_size = int((max_length+20)/1)
    hidden_layer2_size = int((hidden_layer1_size+20)/2)

    model = Sequential()
    model.add(Embedding(input_dim=input_dim, output_dim=output_dim, input_length=max_length, name='embeddings'))
    if option == 'normal':
        model.add(Flatten())
        model.add(Dense(hidden_layer1_size, activation=tf.nn.relu))
        model.add(Dense(hidden_layer2_size, activation=tf.nn.relu))
    elif option == 'lstm':
        model.add(LSTM(hidden_layer1_size, activation=tf.nn.relu))
    model.add(Dense(20, activation=tf.nn.sigmoid))

    optimizer = SGDW(learning_rate=0.05, momentum=0.6, weight_decay=0.0)
    callback = EarlyStopping(monitor='accuracy', patience=3)

    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['binary_crossentropy', 'mean_squared_error', 'accuracy'])

    whole_history = defaultdict(list)
    kf = KFold(n_splits=5, shuffle=True)

    for fold, (train_index, val_index) in enumerate(kf.split(train_vectors)):
        print('--------------FOLD-{}--------------'.format(fold))

        X_train = train_vectors[train_index]
        y_train = train_labels[train_index]
        X_val = train_vectors[val_index]
        y_val = train_labels[val_index]

        history = model.fit(X_train, y_train, epochs=100, batch_size=128, verbose=1, validation_data=(X_val, y_val), callbacks=[callback])
        whole_history = {key: whole_history[key] + value for key, value in history.history.items()}

    results = model.evaluate(test_vectors, test_labels, batch_size=128, verbose=1)
    print('--------------END-OF-TRAINING--------------')
    print('binary_crossentropy:', results[1], 'mean_squared_error:', results[2], 'accuracy:', results[3])
    print('-------------------------------------------')

    plotter(whole_history, 'binary_crossentropy', 'upper right')
    plotter(whole_history, 'mean_squared_error', 'upper right')
    plotter(whole_history, 'accuracy', 'lower right')


def plotter(whole_history, option, loc):
    plt.plot(whole_history[option])
    plt.plot(whole_history['val_' + option])
    plt.title('model ' + option)
    plt.ylabel(option)
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc=loc)
    plt.show()
