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
import os
from sklearn.metrics import precision_score
import matplotlib.pyplot as plt; plt.rcdefaults()
import numpy as np
import matplotlib.pyplot as plt
from multiVectors.utils import createDir
import pickle


# set file directory
arrConfigs=['AI','AN','BI','BA']
fop="../../../../resultETI/d2v/"
createDir(fop)

for idx in range(0,len(arrConfigs)):
    fpInput=fop+arrConfigs[idx]+'_10cv.csv'
    fopOutput=fop+arrConfigs[idx]+"/"
    fpImage = fopOutput + arrConfigs[idx]+'_10cv.png'

    createDir(fopOutput)

    # fnAll='_10cv.csv'
    # load data for 10-fold cv
    df_all = pd.read_csv(fpInput)
    print(list(df_all.columns.values))
    all_label = df_all['score']
    # all_data = df_all.drop(['label','maxSim','maxSim-r2','maxSim-r3','maxSim-r4','maxSim-p1','maxSim-p2','maxSim-p3','maxSim-p4'],axis=1)
    all_data = df_all.drop(['no','score'],axis=1)

    # create a list of classifiers
    random_seed = 2
    classifiers = [GaussianNB(), LogisticRegression(random_state=random_seed),DecisionTreeClassifier(),
                   RandomForestClassifier(random_state=random_seed, n_estimators=50), AdaBoostClassifier(), LinearDiscriminantAnalysis(),QuadraticDiscriminantAnalysis(),
                   LinearSVC(random_state=random_seed), MLPClassifier(alpha=1), GradientBoostingClassifier(random_state=random_seed,  max_depth=5)]

    # fit and evaluate for 10-cv
    index = 0
    # group = df_all['label']
    arrClassifierName = ['Random Forest']
    arrXBar = []
    arrWeightAvg = []
    arrStrWeightAvg = []
    arrIndex=[]
    o2 = open(fopOutput + 'result_all.txt', 'w')
    o2.close()
    k_fold = StratifiedKFold(10,shuffle=True)

    for classifier in classifiers:
        index=index+1
        # filePredict=''.join([fopOutput,'predict_',str(index),'.txt'])
        # print("********", "\n", "10 fold CV Results with: ", str(classifier))
        classifier.fit( all_data, all_label)
        # save the model to disk
        filename4 = fop+arrConfigs[idx]+ '_mlmodel.bin'
        pickle.dump(classifier, open(filename4, 'wb'))


