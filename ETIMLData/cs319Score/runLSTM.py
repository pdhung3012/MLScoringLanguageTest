
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  1 18:19:24 2019

@author: hungphd
"""


# import modules
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.model_selection import cross_val_score,cross_val_predict, StratifiedKFold
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Embedding,SpatialDropout1D,LSTM,Dense
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras_metrics import metrics

# set file directory
fpInput='/home/hung/git/confres/eti/code/step3/cs319Score/distLSTM/'
fnAll='all.csv'
# load data for 10-fold cv
df_all = pd.read_csv(fpInput+fnAll)
print(list(df_all.columns.values))
all_label = df_all['expected']
# all_data = df_all.drop(['label','maxSim','maxSim-r2','maxSim-r3','maxSim-r4','maxSim-p1','maxSim-p2','maxSim-p3','maxSim-p4'],axis=1)
all_data = df_all.drop(['no','reviseNetID','maxSim','predicted','expected'],axis=1)

df_text=df_all['text']
print(df_text)
# The maximum number of words to be used. (most frequent)
MAX_NB_WORDS = 50000
# Max number of words in each complaint.
MAX_SEQUENCE_LENGTH = 250
# This is fixed.
EMBEDDING_DIM = 100
tokenizer = Tokenizer(num_words=MAX_NB_WORDS, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~', lower=True)
tokenizer.fit_on_texts(df_text.values)
word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))

X = tokenizer.texts_to_sequences(df_text.values)
X = pad_sequences(X, maxlen=MAX_SEQUENCE_LENGTH)
# print('Shape of data tensor:', X.shape)

Y = pd.get_dummies(all_label).values
# Y = all_label.values
# print('Shape of label tensor:', str(Y))

# X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.10, random_state = 42)
# print(X_train.shape,Y_train.shape)
# print(X_test.shape,Y_test.shape)

model = Sequential()
model.add(Embedding(MAX_NB_WORDS, EMBEDDING_DIM, input_length=X.shape[1]))
model.add(SpatialDropout1D(0.2))
model.add(LSTM(1000, dropout=0.1, recurrent_dropout=0.1))
model.add(Dense(4, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

epochs = 5
batch_size = 64

# history = model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size,validation_split=0.1,callbacks=[EarlyStopping(monitor='val_loss', patience=3, min_delta=0.0001)])
model.fit(X, Y, validation_split=0.1, epochs=epochs, batch_size=batch_size,callbacks=[EarlyStopping(monitor='val_loss', patience=3, min_delta=0.0001)])
# accr = model.evaluate(X,Y)
# print('Test set\n  Loss: {:0.3f}\n  Accuracy: {:0.3f}'.format(accr[0],accr[1]))
predicted=model.predict(X,verbose=0)
strPredictedPrint=model.predict_classes(X,verbose=0)

matrix = confusion_matrix(Y.argmax(axis=1), predicted.argmax(axis=1))
report = classification_report(Y.argmax(axis=1), predicted.argmax(axis=1))
print(matrix)

o2=open(fpInput+'result_all.txt','w')
o2.write(str(matrix)+'\n')
o2.write(str(report)+'\n')
o2.write(str(strPredictedPrint)+'\n')
o2.close()
# print(str( predicted)+" "+str(Y))
# print(str(classification_report(Y, predicted)))


# k_fold = StratifiedKFold(10,shuffle=True)
# cross_val = cross_val_score(model, X,Y, cv=k_fold, n_jobs=1)
# predicted = cross_val_predict(model, X, Y, cv=k_fold)
# print(str(classification_report(Y, predicted)))
