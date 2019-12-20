from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import cess_esp as cess
from nltk import UnigramTagger as ut

import pandas as pd
from scipy import spatial
import numpy as np
from sklearn.metrics import pairwise_distances
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import cosine
import nltk

def getPOSFromResponse(string):
    # li = list(string.split(" "))
    string = str(string).replace("<p>", "").replace("</p>", "")
    tokens = nltk.word_tokenize(string)
    # str2 = uni_tag.tag(tokens)
    arr =  nltk.pos_tag(tokens)
    str2=""
    for(pos,tag) in arr:
        str2 = str2 + tag + " "
    newStr = str(str2).replace("[", "").replace("]", "").replace("'", "").replace(",", " ").strip()
    return newStr

def readStringFromFile(fpFile):
    f = open(fpFile, "r")
    content=f.read()
    f.close()
    return content
def printVector(vector):
    strOut=""
    i=0
    for item in vector:
        i=i+1
        strOut = strOut+ str(item)
        if(i!= len(vector)):
            strOut=strOut+ ","
    return strOut


def getData(fpInputSubmission, fopTextFolder, fpOutputVectorDistance, fpOutputVectorTFIDF):
    my_csv = pd.read_csv(fpInputSubmission)
    # filtered = my_csv.Score.str.match("I-",na=False)
    # my_csv3 = my_csv2[my_csv2.Score != "I-UR"]
    columnRevisedNetID = my_csv.RevisedNetID
    columnScore = my_csv.Score


    strA = 'A'
    strAMinus = 'A-'
    strBPlus = 'B+'
    strB = 'B'
    strBMinus = 'A-'

    dictScoreResponse = {strA: [], strAMinus: [], strBPlus: [], strB: [], strBMinus: []}

    print(str(len(columnRevisedNetID)))
    i=-1
    listIndexInExcel=[]
    for item in columnRevisedNetID:
        # print(columnScore[i])
        i=i+1
        strScore=columnScore[i]
        fpTextItem=fopTextFolder+str(item)+"/text.txt"
        listIndexInExcel.append(i)
        strResponse = readStringFromFile(fpTextItem)

        if strScore == strA:
            dictScoreResponse[strA].append(strResponse)
        elif strScore == strAMinus:
            dictScoreResponse[strAMinus].append(strResponse)
        elif strScore == strBPlus:
            dictScoreResponse[strBPlus].append(strResponse)
        elif strScore == strB:
            dictScoreResponse[strB].append(strResponse)
        elif strScore == strBMinus:
            dictScoreResponse[strBMinus].append(strResponse)

    strTotalA = ' '.join(dictScoreResponse[strA])
    strTotalAMinus = ' '.join(dictScoreResponse[strAMinus])
    strTotalBPlus = ' '.join(dictScoreResponse[strBPlus])
    strTotalB = ' '.join(dictScoreResponse[strB])
    strTotalBMinus = ' '.join(dictScoreResponse[strBMinus])

    # print(strTotalIMF)
    corpus = [str(strTotalA), str(strTotalAMinus), str(strTotalBPlus), str(strTotalB), str(strTotalBMinus)]

    for i in range(len(columnRevisedNetID)):
        strScore = str(columnScore[i])
        strScore.strip()
        fpTextItem = fopTextFolder + str(columnRevisedNetID[i]) + "/text.txt"
        strResponse = readStringFromFile(fpTextItem)
        corpus.append(strResponse)

    vectorizer = TfidfVectorizer(ngram_range=(1, 4))
    X = vectorizer.fit_transform(corpus)
    arrFeatureNames = vectorizer.get_feature_names()
    print('names: ' + str(len(arrFeatureNames)) + ' ' + str(arrFeatureNames))

    dictTopicVectors = {strA: [], strAMinus: [], strBPlus: [], strB: [], strBMinus: []}

    dictTopicVectors[strA] = X[0].todense()
    dictTopicVectors[strAMinus] = X[1].todense()
    dictTopicVectors[strBPlus] = X[2].todense()
    dictTopicVectors[strB] = X[3].todense()
    dictTopicVectors[strBMinus] = X[4].todense()

    csv = open(fpOutputVectorDistance, 'w')
    csvTFIDF = open(fpOutputVectorTFIDF, 'w')

    columnTitleRow = "no,reviseNetID,distA,distAMinus,distBPlus,distB,distBMinus,maxSim,predicted,expected\n"

    columnTDFTitle ="no,reviseNetID,expected,"

    vector=X[6].toarray()[0]
    print(str(len(vector)))
    print(str(vector[0]))
    for i in range(len(vector)):
        columnTDFTitle = columnTDFTitle+ str((i+1))
        if i!=len(vector)-1:
            columnTDFTitle=columnTDFTitle+','
    columnTDFTitle = columnTDFTitle + '\n'
    csv.write(columnTitleRow)
    csvTFIDF.write(columnTDFTitle)

    print(str(len(corpus)))
    for i in range(5, len(corpus)):
        vectori = X[i].todense()
        strVectorContent=printVector(X[i].toarray()[0])

        distA = cosine_similarity(vectori, dictTopicVectors[strA])[0][0]
        distAMinus = cosine_similarity(vectori, dictTopicVectors[strAMinus])[0][0]
        distBPlus = cosine_similarity(vectori, dictTopicVectors[strBPlus])[0][0]
        distB = cosine_similarity(vectori, dictTopicVectors[strB])[0][0]
        distBMinus = cosine_similarity(vectori, dictTopicVectors[strBMinus])[0][0]


        lst = [distA, distAMinus, distBPlus, distB,distBMinus]
        maxDist = max(lst)

        classResult = strA
        indexCorpus=listIndexInExcel[i-5]
        expectedResult = columnScore[indexCorpus]
        strTestId=columnRevisedNetID[indexCorpus]
        # strTopic=columnTopic[indexCorpus]

        if distA == maxDist:
            classResult = strA
        elif distAMinus == maxDist:
            classResult = strAMinus
        elif distBPlus == maxDist:
            classResult = strBPlus
        elif distB == maxDist:
            classResult = strB
        else:
            classResult = strBMinus
        row = str(i - 4)+ ',' + str(strTestId) + ',' + str(distA) + ',' + str(distAMinus) + ',' + str(distBPlus) + ',' + str(
            distB) + ',' + str(distBMinus) + ',' + str(
            maxDist) + ',' + str(classResult) + ',' + str(expectedResult) + '\n'
        csv.write(row)
        rowTFIDF=str(i - 4)+ ',' + str(strTestId)+',' + str(expectedResult) + ','+strVectorContent+ '\n'
        csvTFIDF.write(rowTFIDF)

def main():
    fopInput="/Users/hungphan/Downloads/hw2VectorData/"
    fopText = "/Users/hungphan/Downloads/hw2TextualInformation/"
    fpStudentSubmission=fopInput+"StudentScore - ReviseList.csv"
    fpStudentDistance = fopInput + "vectorDistance.csv"
    fpStudentTFIDF = fopInput + "vectorTFTDF.csv"
    getData(fpStudentSubmission, fopText,fpStudentDistance, fpStudentTFIDF)

main()
