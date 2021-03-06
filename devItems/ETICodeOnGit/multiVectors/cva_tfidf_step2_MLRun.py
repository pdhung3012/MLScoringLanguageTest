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

# set file directory
nGramNum = 3
arrConfigs=['AI','AN','BI','BA']
fop="../../../../resultETI/cva_tf_idf_"+str(nGramNum)+"/"
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
    all_label = df_all['expected']
    # all_data = df_all.drop(['label','maxSim','maxSim-r2','maxSim-r3','maxSim-r4','maxSim-p1','maxSim-p2','maxSim-p3','maxSim-p4'],axis=1)
    all_data = df_all.drop(['no','testId','topic','maxSim','predicted','expected'],axis=1)

    # create a list of classifiers
    random_seed = 2
    classifiers = [GaussianNB(), LogisticRegression(random_state=random_seed),DecisionTreeClassifier(),
                   RandomForestClassifier(random_state=random_seed, n_estimators=50), AdaBoostClassifier(), LinearDiscriminantAnalysis(),QuadraticDiscriminantAnalysis(),
                   LinearSVC(random_state=random_seed), MLPClassifier(alpha=1), GradientBoostingClassifier(random_state=random_seed,  max_depth=3)]

    # fit and evaluate for 10-cv
    index = 0
    # group = df_all['label']
    arrClassifierName = ['GaussianNB', 'Logistic Regression', 'Decision Tree', 'Random Forest', 'Ada Boost', 'LDA',
                         'QDA', 'Linear SVC', 'MLP', 'Gradient Boosting']
    arrXBar = []
    arrWeightAvg = []
    arrStrWeightAvg = []
    arrIndex=[]
    o2 = open(fopOutput + 'result_all.txt', 'w')
    o2.close()
    k_fold = StratifiedKFold(10,shuffle=True)

    for classifier in classifiers:
                    index=index+1
                    filePredict=''.join([fopOutput,'predict_',str(index),'.txt'])
                    print("********", "\n", "10 fold CV Results with: ", str(classifier))
                    cross_val = cross_val_score(classifier, all_data, all_label, cv=k_fold, n_jobs=1)
                    predicted = cross_val_predict(classifier, all_data, all_label, cv=k_fold)
                    weightAvg = precision_score(all_label, predicted, average='weighted') * 100

                    np.savetxt(filePredict,predicted,fmt='%s', delimiter=',')
                    o2 = open(fopOutput + 'result_all.txt', 'a')
                    o2.write('Result for '+str(classifier)+'\n')
                    o2.write(str(sum(cross_val)/float(len(cross_val)))+'\n')
                    o2.write(str(confusion_matrix(all_label, predicted))+'\n')
                    o2.write(str(classification_report(all_label, predicted))+'\n')
                    o2.close()

                    strClassX =  str(arrClassifierName[index-1])
                    arrIndex.append(index)
                    arrXBar.append(strClassX)
                    arrWeightAvg.append(weightAvg)
                    arrStrWeightAvg.append('{:.2f}'.format(weightAvg))
    # o2.close()

    y_pos = np.arange(len(arrXBar))
    plt.bar(y_pos, arrWeightAvg, align='center', alpha=0.5)
    plt.xticks(y_pos, arrIndex, rotation=90)
    plt.rcParams["figure.figsize"] = (40, 40)
    plt.ylabel('Weighted Precision')
    plt.ylim(0, 100)
    for i in range(len(arrWeightAvg)):
        plt.text(x=i - 0.5, y=arrWeightAvg[i] + 1, s=arrStrWeightAvg[i])
        plt.text(x=i, y=arrWeightAvg[i] - 5, s=arrXBar[i], rotation=90)

    plt.title(fopOutput)
    plt.savefig(fpImage)
    plt.clf()