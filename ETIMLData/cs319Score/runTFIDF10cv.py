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
fpInput='/home/hung/git/confres/eti/code/step3/cs319Score/combine10cv/'
fnAll='all.csv'
# load data for 10-fold cv
df_all = pd.read_csv(fpInput+fnAll)
print(list(df_all.columns.values))
all_label = df_all['expected']
# all_data = df_all.drop(['label','maxSim','maxSim-r2','maxSim-r3','maxSim-r4','maxSim-p1','maxSim-p2','maxSim-p3','maxSim-p4'],axis=1)
all_data = df_all.drop(['no','reviseNetID','expected'],axis=1)

# create a list of classifiers
random_seed = 1234
classifiers = [GaussianNB(), LogisticRegression(random_state=random_seed),DecisionTreeClassifier(),
               RandomForestClassifier(random_state=random_seed, n_estimators=50), AdaBoostClassifier(), LinearDiscriminantAnalysis(),QuadraticDiscriminantAnalysis(),
               LinearSVC(random_state=random_seed)]

# fit and evaluate for 10-cv
index = 0
# group = df_all['label']
o2=open(fpInput+'result_all.txt','w')
k_fold = StratifiedKFold(10,shuffle=True)
for classifier in classifiers:
                index=index+1
                filePredict=''.join([fpInput,'predict_',str(index),'.txt'])
                print("********", "\n", "10 fold CV Results with: ", str(classifier))
                cross_val = cross_val_score(classifier, all_data, all_label, cv=k_fold, n_jobs=1)
                predicted = cross_val_predict(classifier, all_data, all_label, cv=k_fold)
                np.savetxt(filePredict,predicted,fmt='%s', delimiter=',')
                o2.write('Result for '+str(classifier)+'\n')
                o2.write(str(sum(cross_val)/float(len(cross_val)))+'\n')
                o2.write(str(confusion_matrix(all_label, predicted))+'\n')
                o2.write(str(classification_report(all_label, predicted))+'\n')
o2.close()
