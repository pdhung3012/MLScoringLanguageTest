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

def getData(fpInputYear,fpOutputYear):
    my_csv = pd.read_csv(fpInputYear)
    # filtered = my_csv.Score.str.match("I-",na=False)
    # my_csv3 = my_csv2[my_csv2.Score != "I-UR"]
    columnResponse = my_csv.Response
    columnScore = my_csv.Score
    columnTestid = my_csv.Testid
    columnTopic = my_csv.Topic
    columnUserName=my_csv.Username
    columnPOS = my_csv.POS
    listSeparateTopic=my_csv.Topic.unique()
    dictTopicResponse = {}
    dictTopicString = {}

    strMF = 'I-MF'
    strMM = 'I-MM'
    strSE = 'I-SE'
    strNE = 'I-NE'
    dictScoreResponse = {strMF: [], strMM: [], strSE: [], strNE: []}
    print(listSeparateTopic)
    print(str(len(listSeparateTopic)))
    for item in listSeparateTopic:
        dictTopicResponse[str(item)]=[]

    print(str(len(columnResponse)),'\t',str(len(columnScore)))
    i=-1
    listIndexInExcel=[]
    for item in columnResponse:
        # print(columnScore[i])
        i=i+1
        strScore = str(columnScore[i])
        strScore.strip()

        if not (strScore.startswith('I-') and strScore != 'I-UR'):
            continue
        listIndexInExcel.append(i)
        # columnResponse[i]=columnResponse[i].replace("<p>", "").replace("</p>", "")
        # strPOS=getPOSFromResponse(columnResponse[i])
        strResponse = str(columnResponse[i]).replace("<p>", "").replace("</p>", "").replace("<br>", "")
        # print(strResponse)
        strTopic=str(columnTopic[i])

        # if strScore == strMF:
        #     dictScoreResponse[strMF].append(strResponse)
        #     # dictScoreResponse['I-MF'].append(' ')
        #     # dictScoreResponse['I-MF'].append(strPOS)
        # elif strScore == strMM:
        #     dictScoreResponse[strMM].append(strResponse)
        #     # dictScoreResponse['I-MM'].append(' ')
        #     # dictScoreResponse['I-MM'].append(strPOS)
        # elif strScore == strSE:
        #     dictScoreResponse[strSE].append(strResponse)
        #     # dictScoreResponse['I-SE'].append(' ')
        #     # dictScoreResponse['I-SE'].append(strPOS)
        # elif strScore == strNE:
        #     dictScoreResponse[strNE].append(strResponse)
        #     # dictScoreResponse['I-NE'].append(' ')
        #     # dictScoreResponse['I-NE'].append(strPOS)
        dictTopicResponse[strTopic].append(strResponse)


    strTotalIMF = ' '.join(dictScoreResponse[strMF])
    strTotalIMM = ' '.join(dictScoreResponse[strMM])
    strTotalISE = ' '.join(dictScoreResponse[strSE])
    strTotalINE = ' '.join(dictScoreResponse[strNE])
    for item in listSeparateTopic:
        strContentEachTopic =  ' '.join(dictTopicResponse[item]).strip()
        if strContentEachTopic:
            dictTopicString[item]=strContentEachTopic

    numOfLabelTopic=0

    # print(strTotalIMF)
    # corpus = [str(strTotalIMF), str(strTotalIMM), str(strTotalISE), str(strTotalINE)]
    corpus=[]
    for item in dictTopicString:
        corpus.append(str(dictTopicString[item]))
        numOfLabelTopic=numOfLabelTopic+1

    for i in range(len(columnResponse)):
        strScore = str(columnScore[i])
        strScore.strip()
        if not (strScore.startswith('I-') and strScore != 'I-UR'):
            continue
        strResponse = str(columnResponse[i]).replace("<p>", "").replace("</p>", "").replace("<br>", "")
        corpus.append(strResponse)

    vectorizer = TfidfVectorizer(ngram_range=(1, 4))
    X = vectorizer.fit_transform(corpus)
    arrFeatureNames = vectorizer.get_feature_names()
    print('names: ' + str(len(arrFeatureNames)) + ' ' + str(arrFeatureNames))
    dictTopicVectors = {}
    indexNumTopic=0
    columnTitleRow = "no,username,testId,topic,"
    for item in dictTopicString:
        dictTopicVectors[item] = X[indexNumTopic].todense()
        indexNumTopic=indexNumTopic+1
        columnTitleRow=''.join([columnTitleRow,item,","])
    columnTitleRow = ''.join([columnTitleRow, "\n"])
    csv = open(fpOutputYear, 'w')



    csv.write(columnTitleRow)

    print(str(len(corpus)))
    for i in range(numOfLabelTopic, len(corpus)):
        vectori = X[i].todense()
        rowList=""
        for item in dictTopicString:
            distItem = cosine_similarity(vectori, dictTopicVectors[item])[0][0]
            rowList = ''.join([rowList, str(distItem), ","])

        indexCorpus=listIndexInExcel[i-numOfLabelTopic]
        expectedResult = columnTopic[indexCorpus]
        strUsername=columnUserName[indexCorpus]
        strTestId=str(columnTestid[indexCorpus])
        strTopic=str(columnTopic[indexCorpus])
        # strResponse = str(columnResponse[indexCorpus])
        # strPOS = str(columnPOS[indexCorpus])
        row = str(i - numOfLabelTopic+1)+","+strUsername+","+strTestId+","+strTopic+",";
        row = ''.join([row,rowList,  "\n"])
        print(str(len(corpus))+" index "+str(indexCorpus))
        csv.write(row)
        # if(indexCorpus>50):
        #     break
    print(listSeparateTopic)
    print(str(len(listSeparateTopic)))
        # indexItemLabel=0
        # for item in dictTopicString:
        #     row = ''.join([row, str(lst[indexItemLabel]), ","])
        #     indexItemLabel=indexItemLabel+1
        # + ',' + str(strTestId)+ ',' + str(strTopic) + ',' + str(distIMF) + ',' + str(distIMM) + ',' + str(distISE) + ',' + str(
        #     distINE) + ',' + str(
        #     maxDist) + ',' + str(classResult) + ',' + str(expectedResult) + '\n'

    #
    # if distIMF == maxDist:
    #     classResult = strMF
    # elif distIMM == maxDist:
    #     classResult = strMM
    # elif distISE == maxDist:
    #     classResult = strSE
    # else:
    #     classResult = strNE

    # print(
    #     str(i) + '\t' + str(distIMF) + '\t' + str(distIMM) + '\t' + str(distISE) + '\t' + str(distINE) + '\t' + str(
    #         maxDist) + '\t' + str(classResult))
    # lst.append(distItem)

    # distIMM = cosine_similarity(vectori, dictTopicVectors[strMM])[0][0]
    # distISE = cosine_similarity(vectori, dictTopicVectors[strSE])[0][0]
    # distINE = cosine_similarity(vectori, dictTopicVectors[strNE])[0][0]

    # distIMF = cosine_similarity(vectori, vectori)[0][0]
    # distIMM = cosine_similarity(vectori, vectori)[0][0]
    # distISE = cosine_similarity(vectori, vectori)[0][0]
    # distINE = cosine_similarity(vectori, vectori)[0][0]

    # maxDist = max(lst)

    # classResult = strNE

def main():
    fpInputCombine = 'rpf_combine_2018_2019.csv'
    fpOutputCombine = 'vector_combine.csv'
    fpInputNewYear2019 = 'ratingPositionFilter_newData_2019.csv'
    fpOutputNewYear2019 = 'vector_topic_newData_2019.csv'
    getData(fpInputCombine,fpOutputCombine)
    # getData(fpInputNewYear2019, fpOutputNewYear2019)

main()
