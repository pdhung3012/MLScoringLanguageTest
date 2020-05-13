# -*- coding: utf-8 -*-
"""
Created on Fri Mar  1 18:13:25 2019

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
from sklearn.model_selection import cross_val_score,cross_val_predict, StratifiedKFold

# set file directory
fpInput='/Users/ziweizh/Desktop/Kingland/csv/'

# load data for 10-fold cv
df_all = pd.read_csv(fpInput+'all.row.csv')
list(df_all.columns.values)
all_labels = df_all['label']
all_data = df_all.drop(['label','id','doc_id'],axis=1)

# upsample data for 10-fold cv
target_0_all = df_all[df_all['label']==0]
target_1_all = df_all[df_all['label']==1]
target1_upsampled_all = resample(target_1_all,replace=True, n_samples=len(target_0_all), random_state=123)
print(len(target1_upsampled_all))
print(len(target_0_all))
df_full_upsampled_all = pd.concat([target_0_all,target1_upsampled_all])
upsampled_all_label = df_full_upsampled_all['label']
upsampled_all_data = df_full_upsampled_all.drop(['label','id','doc_id'],axis=1)

# create a list of classifiers
random_seed = 1234
classifiers = [GaussianNB(), LogisticRegression(random_state=random_seed),DecisionTreeClassifier(), 
               RandomForestClassifier(random_state=random_seed, n_estimators=50), AdaBoostClassifier(), LinearDiscriminantAnalysis(),QuadraticDiscriminantAnalysis(),
               LinearSVC(random_state=random_seed), MLPClassifier(alpha=1), GradientBoostingClassifier(random_state=random_seed, n_estimators=20, max_depth=5, max_features=20)]

# fit and evaluated for upsampled 10-fold cv
index = 0
group = df_full_upsampled_all['doc_id']
k_fold = StratifiedKFold(10,shuffle=True)
for classifier in classifiers:
                index=index+1
                filePredict=''.join([fpInput,'predict_',str(index),'.txt'])
                o2=open(fpInput+'result_'+str(index)+'.txt','w')
                print("********", "\n", "10 fold CV Results with: ", str(classifier))
                cross_val = cross_val_score(classifier, upsampled_all_data, upsampled_all_label, cv=k_fold, groups=group, n_jobs=1)
                predicted = cross_val_predict(classifier, upsampled_all_data, upsampled_all_label, cv=k_fold)
                np.savetxt(filePredict,predicted, delimiter=',')
                o2.write('Result for '+str(classifier)+'\n')
                o2.write(str(sum(cross_val)/float(len(cross_val)))+'\n')
                o2.write(str(confusion_matrix(upsampled_all_label, predicted))+'\n')
                o2.write(str(classification_report(upsampled_all_label, predicted))+'\n')
                o2.close()