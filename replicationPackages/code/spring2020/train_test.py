# -*- coding: utf-8 -*-
"""
Created on Fri Mar  1 18:31:18 2019

@author: ziweizh
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

# set file directory
fpInput='/Users/ziweizh/Desktop/Kingland/csv/'

# load training and test data
df = pd.read_csv(fpInput+'train.all.csv')
train_label = df['Target']
train_data = df.drop(['ID','Target'],axis=1)
df2 = pd.read_csv(fpInput+'test.all.csv')
test_label = df2['Target']
test_data = df2.drop(['ID','Target'],axis=1)

# create a list of classifiers
random_seed = 1234
classifiers = [GaussianNB(), LogisticRegression(random_state=random_seed),DecisionTreeClassifier(), 
               RandomForestClassifier(random_state=random_seed, n_estimators=50), AdaBoostClassifier(), LinearDiscriminantAnalysis(),QuadraticDiscriminantAnalysis(),
               LinearSVC(random_state=random_seed), MLPClassifier(alpha=1), GradientBoostingClassifier(random_state=random_seed, n_estimators=20, max_depth=5, max_features=20)]

# fit and evaluate
index = 0
for classifier in classifiers:
                index=index+1
                filePredict=''.join([fpInput,'predict_',str(index),'.txt'])
                o2=open(fpInput+'result_'+str(index)+'.txt','w')
                print("********", "\n", "10 fold CV Results with: ", str(classifier))
                classifier.fit(train_data,train_label)                
                predicted = classifier.predict(test_data)                
                np.savetxt(filePredict,predicted, delimiter=',')
                o2.write('Result for '+str(classifier)+'\n')
                o2.write(str(np.mean(predicted == test_label,dtype=float))+'\n')
                o2.write(str(confusion_matrix(test_label, predicted))+'\n')
                o2.write(str(classification_report(test_label, predicted))+'\n')
                o2.close()