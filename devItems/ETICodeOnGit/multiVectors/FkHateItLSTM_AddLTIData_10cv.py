from math import sqrt
from numpy import concatenate
from matplotlib import pyplot
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
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


# from pandas import read_csv
# from datetime import datetime
# # load data
# def parse(x):
# 	return datetime.strptime(x, '%Y %m %d %H')
# dataset = read_csv('raw.csv',  parse_dates = [['year', 'month', 'day', 'hour']], index_col=0, date_parser=parse)
# dataset.drop('No', axis=1, inplace=True)
# # manually specify column names
# dataset.columns = ['pollution', 'dew', 'temp', 'press', 'wnd_dir', 'wnd_spd', 'snow', 'rain']
# dataset.index.name = 'date'
# # mark all NA values with 0
# dataset['pollution'].fillna(0, inplace=True)
# # drop the first 24 hours
# dataset = dataset[24:]
# # summarize first 5 rows
# print(dataset.head(5))
# # save to file
# dataset.to_csv('pollution.csv')

# convert series to supervised learning
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j + 1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j + 1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j + 1, i)) for j in range(n_vars)]
    # put it all together
    agg = concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg


# # load dataset
# dataset = read_csv('pollution.csv', header=0, index_col=0)
# dataset.loc[dataset.pollution <= 5, 'pollution'] = 0 #small
# dataset.loc[(dataset.pollution > 5) & (dataset.pollution <= 15), 'pollution'] = 1 #medium
# dataset.loc[dataset.pollution > 15, 'pollution'] = 2 #very large
# values = dataset.values
# # print('values {}'.format(values))
# # integer encode direction
# encoder = LabelEncoder()
# values[:, 4] = encoder.fit_transform(values[:, 4])
# # ensure all data is float
# values = values.astype('float32')
# # normalize features
# scaler = MinMaxScaler(feature_range=(0, 1))
# scaled = scaler.fit_transform(values)
# # frame as supervised learning
# reframed = series_to_supervised(scaled, 1, 1)
# # drop columns we don't want to predict
# reframed.drop(reframed.columns[[9, 10, 11, 12, 13, 14, 15]], axis=1, inplace=True)
# print(reframed.head())
#
# # split into train and test sets
# values = reframed.values
# n_train_hours = 365 * 24
# train = values[:n_train_hours, :]
# test = values[n_train_hours:, :]
# # split into input and outputs
# train_X, train_y = train[:, :-1], train[:, -1]
# test_X, test_y = test[:, :-1], test[:, -1]



data = read_csv('/home/hung/git/resultETI/d2vAndDeep/vector/BI_d2v_vector.csv')
log_directory='/home/hung/git/resultETI/d2vAndDeep/log/'
createDir(log_directory)
typeName='BI'
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
label=np.asarray(label)
features=data.drop(['no','score'], axis=1)
features=np.asarray(features)
print('fff {} {}'.format(features,label))
train_X, test_X, train_y, test_y = train_test_split(
        features, label, test_size = 0.2, random_state = 42)





# reshape input to be 3D [samples, timesteps, features]
train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
# print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)
print('shape {} {} {}'.format(train_X.shape[0],train_X.shape[1],train_X.shape[2]))
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
kf = KFold(n_splits=10, shuffle=True, random_state=30)
for train_index, test_index in kf.split(label):
    # print("TRAIN:", train_index, "TEST:", test_index)
    train_X, test_X = features[train_index], features[test_index]
    train_y, test_y = label[train_index], label[test_index]
    train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
    test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
    # model = Sequential()
    # model.add(LSTM(50, input_shape=(train_X.shape[1], train_X.shape[2])))
    # model.add(Dense(4, activation='softmax'))
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    history = model.fit(train_X, train_y, validation_split=0.1, epochs=15,verbose=0)
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


# # plot history
# pyplot.plot(history.history['accuracy'], label='train')
# pyplot.plot(history.history['accuracy'], label='test')
# pyplot.legend()
# pyplot.show()
#
# # make a prediction
# yhat = model.predict(test_X)
# print('predict {}'.format(yhat))
# # test_X = test_X.reshape((test_X.shape[0], test_X.shape[2]))
# # # invert scaling for forecast
# # inv_yhat = concatenate((yhat, test_X[:, 1:]), axis=1)
# # inv_yhat = scaler.inverse_transform(inv_yhat)
# # inv_yhat = inv_yhat[:, 0]
# # # invert scaling for actual
# # test_y = test_y.reshape((len(test_y), 1))
# # inv_y = concatenate((test_y, test_X[:, 1:]), axis=1)
# # inv_y = scaler.inverse_transform(inv_y)
# # inv_y = inv_y[:, 0]
# # # calculate RMSE
# # rmse = sqrt(mean_squared_error(inv_y, inv_yhat))
# # print('Test RMSE: %.3f' % rmse)


# kf = KFold(n_splits=10, shuffle=True, random_state=30)
# for train_index, test_index in kf.split(label):
#     # print("TRAIN:", train_index, "TEST:", test_index)
#     train_X_org, test_X_org = features[train_index], features[test_index]
#     train_y, test_y = label[train_index], label[test_index]
#
#     # reshape input to be 3D [samples, timesteps, features]
#     train_X = train_X_org.reshape((train_X_org.shape[0], 1, train_X_org.shape[1]))
#     test_X = test_X_org.reshape((test_X_org.shape[0], 1, test_X_org.shape[1]))
#     # train_X=train_X_org
#     # test_X=test_X_org
#     # print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)
#     print('shape {} {} {}'.format(train_X.shape[0], train_X.shape[1], train_X.shape[2]))
#     # design network
#     model = Sequential()
#     model.add(LSTM(50, input_shape=(train_X_ok.shape[1], train_X_ok.shape[2])))
#     # model.add(Dense(1))
#     # model.compile(loss='mae', optimizer='adam')
#     model.add(Dense(4, activation='softmax'))
#     model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
#     # fit network
#     model.fit(train_X_ok, train_y_ok, epochs=15, batch_size=72, validation_data=(test_X_ok, test_y_ok), verbose=2,
#               shuffle=False)
#
#     y_predict_number = model.predict(test_X)
#     # print('{} aaa {}'.format(y_predict_number,y_test))
#     for index in range(len(y_predict_number)):
#         # print('{}'.format(y_predict_number[index]))
#         listItem = y_predict_number[index].tolist()
#         indexMax = listItem.index(max(listItem))
#         listPredictedLSTM.append(dictLabel[indexMax])
#         listTestLSTM.append(dictLabel[test_y[index]])
