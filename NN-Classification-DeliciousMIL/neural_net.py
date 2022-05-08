import tensorflow as tf
from keras.layers import Dense
from tensorflow_addons.optimizers import SGDW
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import KFold
from collections import defaultdict
import matplotlib.pyplot as plt


class Net(tf.keras.Model):

    def __init__(self, input_layer_size, hidden_layer1_size, hidden_layer2_size, output_layer_size):
        super().__init__()

        self.dense1 = Dense(hidden_layer1_size, input_dim=input_layer_size, activation=tf.nn.relu)
        if hidden_layer2_size:
            self.dense2 = Dense(hidden_layer2_size, activation=tf.nn.relu)
        self.dense3 = Dense(output_layer_size, activation=tf.nn.sigmoid)

    def call(self, inputs):
        x = self.dense1(inputs)
        try:
            x = self.dense2(x)
        except:
            pass
        return self.dense3(x)


def train_nn(train_vectors, train_labels, test_vectors, test_labels, vocab_size):

    input_layer_size = vocab_size
    output_layer_size = 20
    hidden_layer1_size = int((input_layer_size+output_layer_size)/2)
    hidden_layer2_size = int((hidden_layer1_size+output_layer_size)/2)

    optimizer = SGDW(learning_rate=0.05, momentum=0.6, weight_decay=0.1)
    callback = EarlyStopping(monitor='accuracy', patience=3)

    model = Net(input_layer_size, hidden_layer1_size, hidden_layer2_size, output_layer_size)
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
