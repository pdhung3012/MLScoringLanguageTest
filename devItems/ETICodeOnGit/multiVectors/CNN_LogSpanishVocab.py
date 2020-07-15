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
from tokenize import tokenize, untokenize, NUMBER, STRING, NAME, OP
from nltk import word_tokenize

cwd = os.getcwd()
os.chdir(cwd)
# base_directory = cwd
base_directory = '/home/hung/git/resultETI/SpanishRater/'
print(base_directory)

# directory = cwd + '/Data'
# clean_directory = cwd + '/Clean_Data'
# directory = base_directory + '/Data'
clean_directory = base_directory + '/inputCsvs/'
log_directory = base_directory + '/log/'
vocab_directory = base_directory + '/vocab/'

createDir(log_directory)
createDir(vocab_directory)

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

    def get_coefs(word, *arr): return word, np.asarray(arr, dtype='float32')[:300]

    # , encoding = "utf8"
    embeddings_index = dict(get_coefs(*o.split(" ")) for o in open(EMBEDDING_FILE_FILTER))
    # print('{}'.format(embeddings_index['de']))
    setVocabSpanish=set()
    for key in embeddings_index.keys():
        setVocabSpanish.add(key)

    return embeddings_index,setVocabSpanish


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


# Setup Embedding matrix
setVocab=set()
glove_embedding_index,setVocab = load_glove_index()
# print('{}'.format(len(glove_embedding_index)))
# exit(0)

# with open((base_directory + "/DL-Model_Accuracy.txt"), 'a') as model:
#          model.write("\n project || Model || Loss || Accuracy \n")
# print('hello world abcxyz')
# Iterate through datasets applying CNN and Attention models
o2 = open(log_directory + 'all_lstmAndRNN.txt', 'w')
o2.write('')
o2.close()
o2 = open(log_directory + 'all_cnn.txt', 'w')
o2.write('')
o2.close()

setOutOfVocab = set()
setInVocab = set()

listExcelLogSentence=['Type,NumOfAppearInVocab,NumOfWords,Percentage']
listExcelLogTotal=['Type,NumOfAppearInVocab,NumOfWords,Percentage']

for filename in os.listdir(clean_directory):
    print(filename)
    typeName = filename.replace('_text.csv', '')
    df = pd.read_csv(clean_directory + "/" + filename)
    data = df.reset_index()[['response', 'score']]

    # print('{}'.format(data['score']))
    # data.loc[(data.storypoint <= 4), 'storypoints_mod'] = 'low'
    # data.loc[(data.storypoint >13), 'storypoints_mod'] = 'high'
    # data.loc[((data.storypoint <=13) & (data.storypoint > 4)), 'storypoints_mod'] = 'medium'

    # uniqueLabel = list(set(tuple(x) for x in data['score']))
    uniqueLabel = set(data['score'])
    uniqueLabel = sorted(uniqueLabel)
    dictLabel = {}
    for ii in range(len(uniqueLabel)):
        dictLabel[ii] = uniqueLabel[ii]
    print('{}'.format(uniqueLabel))
    data['sp'] = lb.fit_transform(data['score']).tolist()
    data = data[['response', 'sp']]
    enc = [data['sp'][i].index(1) for i in range(df.shape[0])]
    enc = np.asarray(enc)

    # data setup
    tokenizer.fit_on_texts(
        df['response'].astype(str))  # index of words with length vocab size, ordered in length of frequency

    countAppearInVocab = 0
    countAllWord = 0



    for strResponse in df['response']:
        arrTokens=word_tokenize(str(strResponse))
        countItemAppearInVocab = 0
        countItemAllWord = 0


        for tok in arrTokens:
            if tok in setVocab:
                countAppearInVocab=countAppearInVocab+1
                countItemAppearInVocab = countItemAppearInVocab + 1
                setInVocab.add(tok)
            else:
                setOutOfVocab.add(tok)
            countAllWord=countAllWord+1
            countItemAllWord =countItemAllWord+1
        if(countItemAllWord==0):
            scoreItem=0
        else:
            scoreItem=countItemAppearInVocab*1.0/countItemAllWord
        strItemResponse='{},{},{},{}'.format(typeName,countItemAppearInVocab,countItemAllWord,scoreItem)
        listExcelLogSentence.append(strItemResponse)

    if(countAllWord==0):
        scoreTotal=0
    else:
        scoreTotal=countAppearInVocab*1.0/countAllWord
    strItemTotal = '{},{},{},{}'.format(typeName, countAppearInVocab, countAllWord, scoreTotal)
    listExcelLogTotal.append(strItemTotal)

fpLogSentence=open(vocab_directory+'logSentences.csv','w')
fpLogSentence.write('\n'.join(listExcelLogSentence))
fpLogSentence.close()

fpLogTotal=open(vocab_directory+'logTotal.csv','w')
fpLogTotal.write('\n'.join(listExcelLogTotal))
fpLogTotal.close()

setInVocab=sorted(setInVocab)
setOutOfVocab=sorted(setOutOfVocab)

fpLogTotal=open(vocab_directory+'inVocab.txt','w')
fpLogTotal.write('\n'.join(setInVocab))
fpLogTotal.close()

fpLogTotal=open(vocab_directory+'outVocab.txt','w')
fpLogTotal.write('\n'.join(setOutOfVocab))
fpLogTotal.close()















