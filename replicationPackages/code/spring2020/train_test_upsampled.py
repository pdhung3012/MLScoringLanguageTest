# -*- coding: utf-8 -*-
"""
Created on Fri Mar  1 18:47:07 2019

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
from sklearn.utils import resample

# set file directory
fpInput='/Users/ziweizh/Desktop/Kingland/csv/'

# load training and test data
df = pd.read_csv(fpInput+'train.all.csv')
train_label = df['Target']
train_data = df.drop(['ID','Target'],axis=1)
df2 = pd.read_csv(fpInput+'test.all.csv')
test_label = df2['Target']
test_data = df2.drop(['ID','Target'],axis=1)

# upsample data for training
target_0 = df[df['Target']==0]
target_1 = df[df['Target']==1]
target1_upsampled = resample(target_1,replace=True, n_samples=len(target_0), random_state=123)
print(len(target1_upsampled))
print(len(target_0))
df_full_upsampled = pd.concat([target_0,target_1])
upsampled_train_label = df_full_upsampled['Target']
upsampled_train_data = df_full_upsampled.drop(['ID','Target'],axis=1)

# upsample data for test
target_0_test = df2[df2['Target']==0]
target_1_test = df2[df2['Target']==1]
target1_upsampled_test = resample(target_1_test,replace=True, n_samples=len(target_0_test), random_state=123)
print(len(target1_upsampled_test))
print(len(target_0_test))
df2_full_upsampled = pd.concat([target_0_test,target1_upsampled_test])
upsampled_test_label = df2_full_upsampled['Target']
upsampled_test_data = df2_full_upsampled.drop(['ID','Target'],axis=1)

# create a list of classifiers
random_seed = 1234
classifiers = [GaussianNB(), LogisticRegression(random_state=random_seed),DecisionTreeClassifier(), 
               RandomForestClassifier(random_state=random_seed, n_estimators=50), AdaBoostClassifier(), LinearDiscriminantAnalysis(),QuadraticDiscriminantAnalysis(),
               LinearSVC(random_state=random_seed), MLPClassifier(alpha=1), GradientBoostingClassifier(random_state=random_seed, n_estimators=20, max_depth=5, max_features=20)]

# fit and evaluate for umsampled
index = 0
for classifier in classifiers:
                index=index+1
                filePredict=''.join([fpInput,'predict_',str(index),'.txt'])
                o2=open(fpInput+'result_'+str(index)+'.txt','w')
                print("********", "\n", "10 fold CV Results with: ", str(classifier))
                classifier.fit(upsampled_train_data,upsampled_train_label)                
                predicted = classifier.predict(upsampled_test_data)                
                np.savetxt(filePredict,predicted, delimiter=',')
                o2.write('Result for '+str(classifier)+'\n')
                o2.write(str(np.mean(predicted == upsampled_test_label,dtype=float))+'\n')
                o2.write(str(confusion_matrix(upsampled_test_label, predicted))+'\n')
                o2.write(str(classification_report(upsampled_test_label, predicted))+'\n')
                o2.close()