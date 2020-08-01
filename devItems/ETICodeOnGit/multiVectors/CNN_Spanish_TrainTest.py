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
from tensorflow.keras.layers import Dense, Flatten, LSTM, Conv1D, MaxPooling1D, Dropout, Activation, Conv2D, Reshape, MaxPooling2D
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import Bidirectional
from tensorflow.keras import initializers, regularizers,constraints
from tqdm.notebook import tqdm
import tensorflow.keras.backend as K
import tensorflow_probability as tfp
import tensorflow.compat.v1 as tf
#from textblob import TextBlob
import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import LabelBinarizer
import math
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix
from multiVectors.utils import createDir
from sklearn.model_selection import KFold

cwd = os.getcwd()
os.chdir(cwd)
# base_directory = cwd
base_directory = '/home/hung/git/resultETI/SpanishRater/'
print(base_directory)

# directory = cwd + '/Data'
# clean_directory = cwd + '/Clean_Data'
# directory = base_directory + '/Data'
train_directory = base_directory + '/inputCsvs/'
test_directory = base_directory + '/testCsvs/'
log_directory = base_directory + '/log_trainTest/'
createDir(log_directory)
config = tf.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.Session(config=config)
#preprocessing
embed_size = 300
vocab_size = 2000
maxlen = 35
tokenizer = Tokenizer(num_words= vocab_size)
lb = LabelBinarizer() #for one-hot encoding of response


#create glove embeddings    
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

    def get_coefs(word,*arr): return word, np.asarray(arr, dtype='float32')[:300]
    # , encoding = "utf8"
    embeddings_index = dict(get_coefs(*o.split(" ")) for o in open(EMBEDDING_FILE_FILTER))
    # print('{}'.format(embeddings_index['de']))
    return embeddings_index
   
#Create glove embedding matrix for convolutional operations    
def create_glove(word_index,embeddings_index):
    emb_mean,emb_std = -0.005838499,0.48782197
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
            embedding_matrix[i] =  embedding_vector
        else:
            if word.islower():
                # try to get the embedding of word in titlecase if lowercase is not present
                embedding_vector = embeddings_index.get(word.capitalize())
                if embedding_vector is not None: 
                    embedding_matrix[i] = embedding_vector
                else:
                    count_found-=1
            else:
                count_found-=1
    # print("Got embedding for ",count_found," words.")
    # print("{} aaa {} {} ".format(len(embeddings_index),len(word_index),len(embedding_matrix)))
    return embedding_matrix    
        
#Vanilla Convolutional Neural Network Model    
def basic_cnn(embedding_matrix):
    """
    Based on Yoon Kim's idea of CNN for text: https://arxiv.org/pdf/1408.5882.pdf
    """
    filter_sizes = [1,2,3,5]
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
        
#Dot product helper function for attention model
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
    
#Attention Model 
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


    

#Setup Embedding matrix
glove_embedding_index = load_glove_index()
# print('{}'.format(len(glove_embedding_index)))
# exit(0)

# with open((base_directory + "/DL-Model_Accuracy.txt"), 'a') as model:
#          model.write("\n project || Model || Loss || Accuracy \n")
# print('hello world abcxyz')
#Iterate through datasets applying CNN and Attention models
o2 = open(log_directory + 'all_lstmAndRNN.txt', 'w')
o2.write('')
o2.close()
o2 = open(log_directory + 'all_cnn.txt', 'w')
o2.write('')
o2.close()

for filename in os.listdir(train_directory):
    print(filename)
    typeName=filename.replace('_text.csv','')
    df = pd.read_csv(train_directory + "/" + filename)
    testDF=pd.read_csv(test_directory + "/" + typeName+'.csv')
    data = df.reset_index()[['response', 'score']]

    listResponses=[]
    listScore=[]

    for i in range(len(df['score'])):
        listResponses.append(str(df['response'][i]))
        listScore.append(str(df['score'][i]))
    numTrain=len(listResponses)
    for i in range(len(testDF['Score'])):
        listResponses.append(str(testDF['Response'][i]))
        listScore.append(str(testDF['Score'][i]))
        # print('{}\t{}'.format(i,str(df['response'][i])))



    # print('{}'.format(data['score']))
    # data.loc[(data.storypoint <= 4), 'storypoints_mod'] = 'low'
    # data.loc[(data.storypoint >13), 'storypoints_mod'] = 'high'
    # data.loc[((data.storypoint <=13) & (data.storypoint > 4)), 'storypoints_mod'] = 'medium'

    # uniqueLabel = list(set(tuple(x) for x in data['score']))
    uniqueLabel =set(listScore)
    uniqueLabel = sorted(uniqueLabel)
    dictLabel={}
    dictReverseLabel = {}
    for ii in range(len(uniqueLabel)):
        dictLabel[ii]=uniqueLabel[ii]
        dictReverseLabel[uniqueLabel[ii]]=ii
    print('{}'.format(uniqueLabel))

    # listScore = lb.fit_transform(listScore).tolist()
    # data = data[['response', 'sp']]
    # enc = [l[i].index(1) for i in range(df.shape[0])]
    # enc = np.asarray(enc)
    enc=[]
    for i in range(len(listScore)):
        # print(listScore[i])
        enc.append(dictReverseLabel[listScore[i]])


    
    #data setup
    tokenizer.fit_on_texts(listResponses) #index of words with length vocab size, ordered in length of frequency
    sequences = tokenizer.texts_to_sequences(listResponses)
    data_sq = pad_sequences(sequences, maxlen=maxlen)
    # print('{}'.format(tokenizer))
    
    # split = 0.7
    # x_dim = data_sq.shape
    # # x_train, x_test = data_sq[:math.floor(x_dim[0]*0.7),:], data_sq[math.floor(x_dim[0]*0.7):,:]
    # # print('train-test {} {} {}'.format(len(data_sq),len(x_train),len(x_test)))
    # # y_train, y_test = enc[:math.floor(x_dim[0]*0.7),], enc[math.floor(x_dim[0]*0.7):,]
    #
    # kf = KFold(n_splits=10,shuffle=True,random_state=30)
    # kf.get_n_splits(data_sq)
    # print(kf)

    #create embedding glove matrix for words
    emb_mtx = create_glove(tokenizer.word_index, glove_embedding_index)
    # print('{}'.format(len(emb_mtx)))
    cnn_model = basic_cnn(emb_mtx)

    # RNN Attention Model
    att_model = model_lstm_atten(emb_mtx)

    listIndexesCV=[]
    listPredictedCNN = []
    listTestCNN = []
    listPredictedLSTM = []
    listTestLSTM = []
    countFold=0
    # for train_index, test_index in kf.split(data_sq):
        # print("TRAIN:", train_index, "TEST:", test_index)
    # countFold=countFold+1
    # print('begin fold {}'.format(countFold))
    enc = np.asarray(enc)
    data_sq= np.asarray(data_sq)
    print(data_sq)
    x_train, x_test = data_sq[:numTrain-1], data_sq[numTrain:]
    y_train, y_test = enc[:numTrain-1], enc[numTrain:]

    # reshape input to be 3D [samples, timesteps, features]
    # x_train = x_train.reshape((x_train.shape[0], 1, x_train.shape[1]))
    # x_test = x_test.reshape((x_test.shape[0], 1, x_test.shape[1]))
    # print(x_test)

    cnn_model = basic_cnn(emb_mtx)

    # RNN Attention Model
    att_model = model_lstm_atten(emb_mtx)

    # for item in test_index:
    #     listIndexesCV.append(item)


    # CNN Model training
    # cnn_model.fit(x_train, y_train, validation_split=0.1, epochs=50,verbose=0)
    cnn_model.fit(x_train, y_train, epochs=50, batch_size=72, validation_data=(x_test, y_test), verbose=0,
                  shuffle=False)
    y_predict_number = cnn_model.predict(x_test)
    for index in range(len(y_predict_number)):
        # print('{}'.format(y_predict_number[index]))
        listItem = y_predict_number[index].tolist()
        indexMax = listItem.index(max(listItem))
        listPredictedCNN.append(dictLabel[indexMax])
        listTestCNN.append(dictLabel[y_test[index]])

    #LSTM and attention
    # att_model.fit(x_train, y_train, validation_split=0.1, epochs=15,verbose=0)
    att_model.fit(x_train, y_train, epochs=50, batch_size=72, validation_data=(x_test, y_test), verbose=0,shuffle=False)
    loss, acc = att_model.evaluate(x_test, y_test)

    y_predict_number = att_model.predict(x_test)
    # print('{} aaa {}'.format(y_predict_number,y_test))
    for index in range(len(y_predict_number)):
        # print('{}'.format(y_predict_number[index]))
        listItem = y_predict_number[index].tolist()
        indexMax = listItem.index(max(listItem))
        listPredictedLSTM.append(dictLabel[indexMax])
        listTestLSTM.append(dictLabel[y_test[index]])
        # break

    # x_train, x_test, y_train, y_test = train_test_split(
    #     data_sq, enc, test_size = 0.2, random_state = 42)
    # print('{} aaa {}'.format(y_predict_number,y_test))
    # np.savetxt(log_directory + typeName + '_index.txt', listIndexesCV, fmt='%s', delimiter=',')
    np.savetxt(log_directory+typeName+ '_cnn_predicted.txt', listPredictedCNN, fmt='%s', delimiter=',')
    np.savetxt(log_directory + typeName + '_cnn_test.txt', listTestCNN, fmt='%s', delimiter=',')
    o2 = open(log_directory + 'all_cnn.txt', 'a')
    o2.write('Result for ' + str(typeName) + '\n')
    # o2.write(str(sum(cross_val) / float(len(cross_val))) + '\n')
    o2.write(str(confusion_matrix(listTestCNN, listPredictedCNN)) + '\n')
    o2.write(str(classification_report(listTestCNN, listPredictedCNN)) + '\n')
    o2.close()
    # print('abc {}'.format(enc))
    # print("predictions shape:", y_predict.shape)
    # loss, acc = cnn_model.evaluate(x_test, y_test)
    # with open((base_directory + "/DL-Model_Accuracy.txt"), 'a') as model:
    #          model.write(filename + " Basic-CNN-Model " + str(loss) + " " + str(acc) +"\n")
    # np.savetxt(log_directory + typeName + '_lstmAndRNN_predicted.txt', listPredictedLSTM, fmt='%s', delimiter=',')
    np.savetxt(log_directory + typeName + '_lstmAndRNN_test.txt', listTestLSTM, fmt='%s', delimiter=',')
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

    
'''
References on implementation for glove embeddings and models from:

#https://github.com/dennybritz/cnn-text-classification-tf
https://mlwhiz.com/blog/2019/03/09/deeplearning_architectures_text_classification/
#http://www.wildml.com/2015/11/understanding-convolutional-neural-networks-for-nlp/
#https://medium.com/@sabber/classifying-yelp-review-comments-using-lstm-and-word-embeddings-part-1-eb2275e4066b
#https://towardsdatascience.com/how-to-build-a-gated-convolutional-neural-network-gcnn-for-natural-language-processing-nlp-5ba3ee730bfb
#http://www.wildml.com/2015/12/implementing-a-cnn-for-text-classification-in-tensorflow/
'''
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
