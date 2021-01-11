# -*- coding: utf-8 -*-
"""
Attention model code is based from https://www.kaggle.com/sermakarevich/hierarchical-attention-network
and is based on Yang et al. 2016 paper titled "Hierarchical Attention Networks for Document Classification"
@author: Hung
"""
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import Input
from tensorflow.keras.layers import MaxPool2D, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Layer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, LSTM, Conv1D, MaxPooling1D, Dropout, Activation, Conv2D, Reshape, \
    MaxPooling2D
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import Bidirectional
from tensorflow.keras import initializers, regularizers, constraints
from tqdm.notebook import tqdm
import tensorflow.keras.backend as K
import tensorflow.compat.v1 as tf
# from textblob import TextBlob
import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import LabelBinarizer
import math
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from multiVectors.utils import createDir
from sklearn.model_selection import KFold
from tensorflow import keras
from tensorflow.keras import layers

cwd = os.getcwd()
os.chdir(cwd)
# base_directory = cwd
base_directory = '/home/hung/git/resultETI/d2vAndDeep/'
print(base_directory)

# directory = cwd + '/Data'
# clean_directory = cwd + '/Clean_Data'
# directory = base_directory + '/Data'
clean_directory = base_directory + '/vector/'
log_directory = base_directory + '/log/'
createDir(log_directory)
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
# preprocessing
embed_size = 300
vocab_size = 2000
maxlen = 35
tokenizer = Tokenizer(num_words=vocab_size)
lb = LabelBinarizer()  # for one-hot encoding of response


# create glove embeddings
def load_glove_index():
    EMBEDDING_FILE = base_directory + '../pretrainedGlove/glove-sbwc.i25.vec'
    EMBEDDING_FILE_FILTER = base_directory + '../pretrainedGlove/glove-sbwc.i25_filter.vec'

    # Using readlines()
    # file1 = open(EMBEDDING_FILE, 'r')
    # Lines = file1.readlines()
    # file1.close()
    # Lines.pop(0)
    # print('len {}'.format(len(Lines)))
    # file1 = open(EMBEDDING_FILE_FILTER, 'w')
    # count=0
    # for count in range(1,len(Lines)):
    #     line=Lines[count]
    #     file1.write('{}'.format(line))
    #     # arrItem=line.split(' ')
    #     # print("Line{}: {}".format(count, len(arrItem)))
    # file1.close()

    def get_coefs(word, *arr): return word, np.asarray(arr, dtype='float32')[:300]

    # , encoding = "utf8"
    embeddings_index = dict(get_coefs(*o.split(" ")) for o in open(EMBEDDING_FILE_FILTER))
    # print('{}'.format(embeddings_index['de']))
    return embeddings_index


# Create glove embedding matrix for convolutional operations
def create_glove(word_index, embeddings_index):
    emb_mean, emb_std = -0.005838499, 0.48782197
    # print('len embedding index {} {}'.format(len(embeddings_index),len(embeddings_index.values())))
    all_embs = np.stack(embeddings_index.values())
    embed_size = all_embs.shape[1]
    nb_words = min(vocab_size, len(word_index))
    embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, embed_size))

    count_found = nb_words
    for word, i in tqdm(word_index.items()):
        if i >= vocab_size: continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
        else:
            if word.islower():
                # try to get the embedding of word in titlecase if lowercase is not present
                embedding_vector = embeddings_index.get(word.capitalize())
                if embedding_vector is not None:
                    embedding_matrix[i] = embedding_vector
                else:
                    count_found -= 1
            else:
                count_found -= 1
    # print("Got embedding for ",count_found," words.")
    # print("{} aaa {} {} ".format(len(embeddings_index),len(word_index),len(embedding_matrix)))
    return embedding_matrix


# Vanilla Convolutional Neural Network Model
def basic_cnn(embedding_matrix):
    """
    Based on Yoon Kim's idea of CNN for text: https://arxiv.org/pdf/1408.5882.pdf
    """
    filter_sizes = [1, 2, 3, 5]
    num_filters = 36

    inp = Input(shape=(maxlen,))
    x = Embedding(vocab_size, embed_size, weights=[embedding_matrix])(inp)
    x = Reshape((maxlen, embed_size, 1))(x)

    maxpool_pool = []
    for i in range(len(filter_sizes)):
        conv = Conv2D(num_filters, kernel_size=(filter_sizes[i], embed_size),
                      kernel_initializer='he_normal', activation='relu')(x)
        maxpool_pool.append(MaxPool2D(pool_size=(maxlen - filter_sizes[i] + 1, 1))(conv))

    z = Concatenate(axis=1)(maxpool_pool)
    z = Flatten()(z)
    z = Dropout(0.1)(z)

    outp = Dense(4, activation="softmax")(z)

    model = Model(inputs=inp, outputs=outp)
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model
from numpy import array
# split a univariate sequence into samples
def split_sequence(sequence, n_steps):
	X, y = list(), list()
	for i in range(len(sequence)):
		# find the end of this pattern
		end_ix = i + n_steps
		# check if we are beyond the sequence
		if end_ix > len(sequence)-1:
			break
		# gather input and output parts of the pattern
		seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
		X.append(seq_x)
		y.append(seq_y)
	return array(X), array(y)

def basic_cnn_sequential(trainX,trainy):
    """
    Based on Yoon Kim's idea of CNN for text: https://arxiv.org/pdf/1408.5882.pdf
    """
    # filter_sizes = [1, 2, 3, 5]
    # num_filters = 36
    #
    # inp = Input(shape=(maxlen,))
    # x = Embedding(vocab_size, embed_size, weights=[embedding_matrix])(inp)
    # x = Reshape((maxlen, embed_size, 1))(x)
    #
    # maxpool_pool = []
    # for i in range(len(filter_sizes)):
    #     conv = Conv2D(num_filters, kernel_size=(filter_sizes[i], embed_size),
    #                   kernel_initializer='he_normal', activation='relu')(x)
    #     maxpool_pool.append(MaxPool2D(pool_size=(maxlen - filter_sizes[i] + 1, 1))(conv))
    #
    # z = Concatenate(axis=1)(maxpool_pool)
    # z = Flatten()(z)
    # z = Dropout(0.1)(z)
    #
    # outp = Dense(4, activation="softmax")(z)
    #
    # model = Model(inputs=inp, outputs=outp)
    # model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    # Create the model

    # https://www.machinecurve.com/index.php/2019/09/17/how-to-create-a-cnn-classifier-with-keras/
    # https://machinelearningmastery.com/how-to-develop-convolutional-neural-network-models-for-time-series-forecasting/
    # choose a number of time steps
    n_steps = 1
    # split into samples
    X, y = split_sequence(x_train, n_steps)
    # reshape from [samples, timesteps] into [samples, timesteps, features]
    n_features = 300
    X = trainX.reshape((trainX.shape[0], trainX.shape[1], n_features))
    # define model
    model = Sequential()
    model.add(Conv1D(filters=64, kernel_size=2, activation='relu', input_shape=(n_steps, n_features)))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(50, activation='relu'))
    model.add(Dense(4, activation='softmax'))
    # model.compile(optimizer='adam', loss='mse')
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


# Dot product helper function for attention model
def dot_product(x, kernel):
    """
    Wrapper for dot product operation, in order to be compatible with both
    Theano and Tensorflow
    Args:
        x (): input
        kernel (): weights
    Returns:
    """
    if K.backend() == 'tensorflow':
        return K.squeeze(K.dot(x, K.expand_dims(kernel)), axis=-1)
    else:
        return K.dot(x, kernel)


# Attention Model
class AttentionWithContext(Layer):
    """
    Follows the work of Yang et al. [https://www.cs.cmu.edu/~diyiy/docs/naacl16.pdf]
    "Hierarchical Attention Networks for Document Classification"
    """

    def __init__(self,
                 W_regularizer=None, u_regularizer=None, b_regularizer=None,
                 W_constraint=None, u_constraint=None, b_constraint=None,
                 bias=True, **kwargs):

        self.supports_masking = True
        self.init = initializers.get('glorot_uniform')

        self.W_regularizer = regularizers.get(W_regularizer)
        self.u_regularizer = regularizers.get(u_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.u_constraint = constraints.get(u_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias
        super(AttentionWithContext, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3

        self.W = self.add_weight(shape=(input_shape[-1], input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        if self.bias:
            self.b = self.add_weight(shape=(input_shape[-1],),
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)

        self.u = self.add_weight(shape=(input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_u'.format(self.name),
                                 regularizer=self.u_regularizer,
                                 constraint=self.u_constraint)

        super(AttentionWithContext, self).build(input_shape)

    def compute_mask(self, input, input_mask=None):
        # do not pass the mask to the next layers
        return None

    def call(self, x, mask=None):
        uit = dot_product(x, self.W)
        if self.bias:
            uit += self.b
        uit = K.tanh(uit)
        ait = dot_product(uit, self.u)
        a = K.exp(ait)
        # apply mask after the exp. will be re-normalized next
        if mask is not None:
            # Cast the mask to floatX to avoid float64 upcasting in theano
            a *= K.cast(mask, K.floatx())
        # in some cases especially in the early stages of training the sum may be almost zero
        # and this results in NaN's. A workaround is to add a very small positive number ε to the sum.
        # a /= K.cast(K.sum(a, axis=1, keepdims=True), K.floatx())
        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())

        a = K.expand_dims(a)
        weighted_input = x * a
        return K.sum(weighted_input, axis=1)

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[-1]


def model_lstm_atten(embedding_matrix):
    inp = Input(shape=(maxlen,))
    x = Embedding(vocab_size, embed_size, weights=[embedding_matrix], trainable=False)(inp)
    x = Bidirectional(LSTM(128, return_sequences=True))(x)
    x = Bidirectional(LSTM(64, return_sequences=True))(x)
    x = AttentionWithContext()(x)
    x = Dense(64, activation="relu")(x)
    x = Dense(4, activation="softmax")(x)
    model = Model(inputs=inp, outputs=x)
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def model_lstm_atten_sequential(trainX):
    # inp = Input(shape=(maxlen,))
    # x = Embedding(vocab_size, embed_size, weights=[embedding_matrix], trainable=False)(inp)
    # x = Bidirectional(LSTM(128, return_sequences=True))(x)
    # x = Bidirectional(LSTM(64, return_sequences=True))(x)
    # x = AttentionWithContext()(x)
    # x = Dense(64, activation="relu")(x)
    # x = Dense(4, activation="softmax")(x)
    # model = Model(inputs=inp, outputs=x)
    # model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    # model = keras.Sequential()
    # # Add an Embedding layer expecting input vocab of size 1000, and
    # # output embedding dimension of size 64.
    # model.add(layers.Embedding(input_dim=300, output_dim=4))
    # # Add a LSTM layer with 128 internal units.
    # model.add(layers.LSTM(128))
    # model.add(layers.LSTM(64))
    # # Add a Dense layer with 10 units.
    # model.add(layers.Dense(64,activation='relu'))
    # model.add(layers.Dense(4, activation='softmax'))

    data = trainX.reshape(1, len(trainX), 20)
    # define LSTM
    model = Sequential()
    model.add(LSTM(20, input_shape=(len(trainX), 20)))
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    # model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


# Setup Embedding matrix
glove_embedding_index = load_glove_index()
# print('{}'.format(len(glove_embedding_index)))
# exit(0)

# with open((base_directory + "/DL-Model_Accuracy.txt"), 'a') as model:
#          model.write("\n project || Model || Loss || Accuracy \n")
# print('hello world abcxyz')
# Iterate through datasets applying CNN and Attention models
o2 = open(log_directory + 'all_lstmAndRNN.txt', 'w')
o2.write('')
o2.close()

for filename in os.listdir(clean_directory):
    print(filename)
    typeName = filename.replace('_d2v_vector.csv', '')
    data = pd.read_csv(clean_directory + "/" + filename)
    createDir(log_directory)
    # typeName = 'BI'
    # data = df.reset_index()[['response', 'score']]
    uniqueLabel = set(data['score'])
    uniqueLabel = sorted(uniqueLabel)
    dictLabel = {}
    for ii in range(len(uniqueLabel)):
        dictLabel[ii] = uniqueLabel[ii]
    print('{}'.format(uniqueLabel))
    lb = LabelBinarizer()  # for one-hot encoding of response

    data['score'] = lb.fit_transform(data['score']).tolist()
    label = [data['score'][i].index(1) for i in range(data.shape[0])]
    label = np.asarray(label)
    features = data.drop(['no', 'score'], axis=1)
    features = np.asarray(features)
    print('fff {} {}'.format(features, label))
    train_X, test_X, train_y, test_y = train_test_split(
        features, label, test_size=0.2, random_state=42)

    # reshape input to be 3D [samples, timesteps, features]
    train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
    test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
    print(test_X)
    # print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)
    print('shape {} {} {}'.format(test_X.shape[0], test_X.shape[1], test_X.shape[2]))
    # design network
    model = Sequential()
    model.add(LSTM(50, input_shape=(train_X.shape[1], train_X.shape[2])))
    model.add(Dense(4, activation='softmax'))
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    # # fit network
    # history = model.fit(train_X, train_y, epochs=50, batch_size=72, validation_data=(test_X, test_y), verbose=2,
    #                     shuffle=False)
    # y_predict_number = model.predict(test_X)
    # # print('{} aaa {}'.format(y_predict_number,y_test))
    # listPredictedLSTM=[]
    # listTestLSTM=[]
    # for index in range(len(y_predict_number)):
    #     # print('{}'.format(y_predict_number[index]))
    #     listItem = y_predict_number[index].tolist()
    #     indexMax = listItem.index(max(listItem))
    #     listPredictedLSTM.append(dictLabel[indexMax])
    #     listTestLSTM.append(dictLabel[test_y[index]])

    listPredictedLSTM = []
    listTestLSTM = []
    countFold = 0
    kf = KFold(n_splits=10, shuffle=True, random_state=30)
    for train_index, test_index in kf.split(label):
        # print("TRAIN:", train_index, "TEST:", test_index)
        countFold = countFold + 1
        print('begin fold {}'.format(countFold))
        train_X, test_X = features[train_index], features[test_index]
        train_y, test_y = label[train_index], label[test_index]
        train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
        test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
        model = Sequential()
        model.add(LSTM(50, input_shape=(train_X.shape[1], train_X.shape[2])))
        model.add(Dense(4, activation='softmax'))
        model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        history = model.fit(train_X, train_y, validation_split=0.1, epochs=50,verbose=0)
        y_predict_number = model.predict(test_X)
        for index in range(len(y_predict_number)):
            # print('{}'.format(y_predict_number[index]))
            listItem = y_predict_number[index].tolist()
            indexMax = listItem.index(max(listItem))
            listPredictedLSTM.append(dictLabel[indexMax])
            listTestLSTM.append(dictLabel[test_y[index]])

    np.savetxt(log_directory + typeName + '_d2vAndDeep_predicted.txt', listPredictedLSTM, fmt='%s', delimiter=',')
    np.savetxt(log_directory + typeName + '_d2vAndDeep_test.txt', listTestLSTM, fmt='%s', delimiter=',')
    o2 = open(log_directory + 'all_lstmAndRNN.txt', 'a')
    o2.write('Result for ' + str(typeName) + '\n')
    # o2.write(str(sum(cross_val) / float(len(cross_val))) + '\n')
    o2.write(str(confusion_matrix(listTestLSTM, listPredictedLSTM)) + '\n')
    o2.write(str(classification_report(listTestLSTM, listPredictedLSTM)) + '\n')
    o2.close()
    #
    # with open((base_directory + "/DL-Model_Accuracy.txt"), 'a') as model:
    #          model.write(filename + " LSTM-Attention " + str(loss) + " " + str(acc) +"\n")
    # break
# with open((base_directory + "/DL-Model_Accuracy.txt"), 'a'):
#     pass
















