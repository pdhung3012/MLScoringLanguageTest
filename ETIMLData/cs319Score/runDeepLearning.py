
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
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
import numpy
# fix random seed for reproducibility
numpy.random.seed(7)

# set file directory
fpInput='/home/hung/git/confres/eti/code/step3/cs319Score/distLSTM/'
fnAll='all.csv'
# load data for 10-fold cv
df_all = pd.read_csv(fpInput+fnAll)
print(list(df_all.columns.values))
Y = df_all['expected']
# all_data = df_all.drop(['label','maxSim','maxSim-r2','maxSim-r3','maxSim-r4','maxSim-p1','maxSim-p2','maxSim-p3','maxSim-p4'],axis=1)
all_data = df_all.drop(['no','reviseNetID','text','maxSim','predicted','expected'],axis=1).astype(float)

# encode class values as integers
encoder = LabelEncoder()
encoder.fit(Y)
encoded_Y = encoder.transform(Y)
# convert integers to dummy variables (i.e. one hot encoded)
dummy_y = np_utils.to_categorical(encoded_Y)
# print(dummy_y)

# define baseline model
def baseline_model():
	# create model
	model = Sequential()
	model.add(Dense(8, input_dim=4, activation='relu'))
	model.add(Dense(4, activation='softmax'))
	# Compile model
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model

# model = Sequential()
# model.add(Dense(8, input_dim=4, activation='relu'))
# model.add(Dense(4, activation='softmax'))
# # Compile model
# model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# Fit the model
# model.fit(all_data, all_label, validation_split=0.01, epochs=1500, batch_size=10)
# # accr = model.evaluate(X,Y)
# # print('Test set\n  Loss: {:0.3f}\n  Accuracy: {:0.3f}'.format(accr[0],accr[1]))
# predicted=model.predict_classes(all_data,verbose=0)
# # strPredictedPrint=model.predict_classes(X,verbose=0)
# print(predicted)
# matrix = confusion_matrix(all_label, predicted)
# # report = classification_report(all_label, predicted)
# print(matrix)
#
# o2=open(fpInput+'result_all.txt','w')
# o2.write(str(matrix)+'\n')
# o2.write(str(report)+'\n')
# o2.write(str(predicted)+'\n')
# o2.close()
estimator = KerasClassifier(build_fn=baseline_model, epochs=2000, batch_size=10, verbose=0)
estimator.fit(all_data,dummy_y)
kfold = KFold(n_splits=10, shuffle=True)
results = cross_val_predict(estimator, all_data, dummy_y, cv=kfold)
print(results)
# print("Baseline: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))

# prediction = estimator.predict(all_data)
# print(prediction)
