import flask
from flask import request, jsonify
from base64 import b64encode,b64decode
import json
import traceback

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
import pickle
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.tokenize import word_tokenize
import traceback

import os

def createDir(folderName):
    try:
        # Create target Directory
        os.mkdir(folderName)
        print("Directory ", folderName, " Created ")
    except FileExistsError:
        print("Directory ", folderName, " already exists")
def readFile(fp):
    strResult=''
    try:
        file=open(fp,'r')
        strResult=file.read()
        file.close()
    except:
        print("File ", fp, " doesn't exist")
    return strResult


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


responses = [
    {'testId': '1',
     'content': 'Hola',
     'form': 'A',
     'level': 'I',
     'promptId': '11'},
    {'testId': '2',
     'content': 'Hola 2',
     'form': 'A',
     'level': 'N',
     'promptId': '12'},
    {'testId': '3',
     'content': 'Hola 3',
     'form': 'B',
     'level': 'I',
     'promptId': '13'},
    {'testId': '4',
     'content': 'Hola 4',
     'form': 'B',
     'level': 'A',
     'promptId': '14'}
]

app = flask.Flask(__name__)
app.config["DEBUG"] = True


# strResponse = json.dumps(responses)
# print(strResponse)
# strEncode=b64encode(strResponse.encode('utf-8'))
# print(strEncode)
# strDecode=b64decode(strEncode).decode('utf-8')
# print(strDecode)
#
# responsesObject=json.loads(strDecode)
# print(str(responsesObject))

fopModelLocation="../../../../resultETI/d2v/"
import nltk
nltk.download('punkt')

@app.route('/', methods=['GET'])
def home():
    return '''<h1>LTI</h1>
<p>LTI request.</p>'''
#
#
# @app.route('/api/v1/resources/books/all', methods=['GET'])
# def api_all():
#     return jsonify(books)
#
#
@app.route('/responses', methods=['POST', 'GET'])
def api_jsonData():
    # Check if an ID was provided as part of the URL.
    # If ID is provided, assign it to a variable.
    # If no ID is provided, display an error in the browser.
    try:
        if 'jsonData' in request.args:
            strJson = str(request.args['jsonData'])
            # strDecode = b64decode(strJson).decode('utf-8')
            responseArr = json.loads(strJson)


        else:
            return "Error: No id field provided. Please specify an id."

        # results = []
        # for responseItem in responseArr:
        #     responseItem['score']='MF'
        #     results.append(responseItem)
        results=predictScore(responseArr,fopModelLocation)
    except Exception as e:
        string_error = traceback.format_exc()
        print(string_error)
        return string_error


    # Use the jsonify function from Flask to convert our list of
    # Python dictionaries to the JSON format.
    return jsonify(results)

# result=predictScore(responses,fopModelLocation)


app.run(host='209.124.64.139',port=5000)