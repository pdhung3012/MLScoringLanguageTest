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

# set file directory
fpInput='/home/hung/git/confres/eti/code/step3/rtNew2019TrainTest/intermediate/'
fn2018='train - Sheet1.csv'
fn2019='test - Sheet1.csv'
# load data for 10-fold cv
df_all = pd.read_csv(fpInput+fn2018)
print(list(df_all.columns.values))
all_label = df_all['expected']
# all_data = df_all.drop(['label','maxSim','maxSim-r2','maxSim-r3','maxSim-r4','maxSim-p1','maxSim-p2','maxSim-p3','maxSim-p4'],axis=1)
all_data = df_all.drop(['no','testId','topic','expected','predicted','maxSim'],axis=1)

df_test = pd.read_csv(fpInput+fn2019)
test_label = df_test['expected']
# all_data = df_all.drop(['label','maxSim','maxSim-r2','maxSim-r3','maxSim-r4','maxSim-p1','maxSim-p2','maxSim-p3','maxSim-p4'],axis=1)
test_data = df_test.drop(['no','testId','topic','expected','predicted','maxSim'],axis=1)


# create a list of classifiers
random_seed = 1234
classifiers = [GaussianNB(), LogisticRegression(random_state=random_seed),DecisionTreeClassifier(),
               RandomForestClassifier(random_state=random_seed, n_estimators=50), AdaBoostClassifier(), LinearDiscriminantAnalysis(),QuadraticDiscriminantAnalysis(),
               LinearSVC(random_state=random_seed), MLPClassifier(alpha=1), GradientBoostingClassifier(random_state=random_seed,  max_depth=5)]

index = 0
# group = df_all['label']
# k_fold = StratifiedKFold(10,shuffle=True)
o2=open(fpInput+'result_all.txt','w')
for classifier in classifiers:
                index=index+1
                filePredict=''.join([fpInput,'predict_',str(index),'.txt'])
                # o2=open(fpInput+'result_all.txt','w')
                # print("********", "\n", "10 fold CV Results with: ", str(classifier))
                classifier.fit(all_data, all_label)
                predicted =classifier.predict(test_data)
                # cross_val = cross_val_score(classifier, all_data, all_label, cv=k_fold, n_jobs=1)
                # predicted = cross_val_predict(classifier, all_data, all_label, cv=k_fold)
                np.savetxt(filePredict,predicted,fmt='%s', delimiter=',')
                o2.write('Result for '+str(classifier)+'\n')
                # o2.write(str(sum(cross_val)/float(len(cross_val)))+'\n')
                o2.write(str(confusion_matrix(test_label, predicted))+'\n')
                o2.write(str(classification_report(test_label, predicted))+'\n')

o2.close()
