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
from multiVectors.utils import createDir
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.tokenize import word_tokenize
import traceback


responses = [
    {'testId': '5',
     'content': 'Hola 5',
     'form': 'A',
     'level': 'I',
     'promptId': '11'},
    {'testId': '6',
     'content': 'Hola 6',
     'form': 'A',
     'level': 'N',
     'promptId': '12'}
]

def predictScore(listResponses,fopModelLocation):
    result=listResponses
    try:
        arrConfigs=['AI','AN','BI','BA']
        dictD2VModels={}
        dictMLModels = {}
        for item in arrConfigs:
            fpModelData=fopModelLocation+item+'_10cv.d2v.txt'
            fpMLModel = fopModelLocation + item + '_mlmodel.bin'
            modelD2v = Doc2Vec.load(fpModelData)
            modelML = pickle.load(open(fpMLModel, 'rb'))
            dictD2VModels[item]=modelD2v
            dictMLModels[item]=modelML

        for i in  range(0,len(result)):
            item = result[i]
            try:
                strModelType=item['form']+item['level']
                modelD2v = dictD2VModels[strModelType]
                modelML = dictMLModels[strModelType]
                strContent=item['content']
                x_data = word_tokenize(strContent)
                v1 = modelD2v.infer_vector(x_data)
                arrTestData=[]
                arrTestData.append(v1)
                scoreItem=modelML.predict(arrTestData)
                item['score']=scoreItem[0]
                # print(scoreItem)
                result[i]=item
            except Exception as e:
                item['score'] = 'UR'
                result[i] = item
                string_error = traceback.format_exc()
                print(string_error)

    except Exception as e:
        string_error = traceback.format_exc()
        print(string_error)
    return result


fopModelLocation="../../../../resultETI/d2v/"
result=predictScore(responses,fopModelLocation)
print(str(result))
