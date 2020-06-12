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
import os
# def getPOSFromResponse(string):
#     # li = list(string.split(" "))
#     string = str(string).replace("<p>", "").replace("</p>", "")
#     tokens = nltk.word_tokenize(string)
#     # str2 = uni_tag.tag(tokens)
#     arr =  nltk.pos_tag(tokens)
#     str2=""
#     for(pos,tag) in arr:
#         str2 = str2 + tag + " "
#     newStr = str(str2).replace("[", "").replace("]", "").replace("'", "").replace(",", " ").strip()
#     return newStr

def getDataByNGram(fpInputYear,fpOutputYear,prefix,nGram):

    my_csv = pd.read_csv(fpInputYear)
    my_csv = my_csv.drop(my_csv[~my_csv.Score.str.startswith(prefix, na=False)].index)
    my_csv = my_csv.drop(my_csv[my_csv.Score.str.endswith('UR')].index)
    my_csv = my_csv.reset_index(drop=True)

    # columnPOS = my_csv.POS
    columnScore = my_csv.Score
    columnTestid = my_csv.Testid
    columnTopic = my_csv.Topic
    columnResponse = my_csv.Response

    strMF = prefix+'MF'
    strMM = prefix+'MM'
    strSE = prefix+'SE'
    strNE = prefix+'NE'
    strUR = prefix+'UR'
    dictScoreResponse = {strMF: [], strMM: [], strSE: [], strNE: []}

    # print(str(len(columnPOS)),'\t',str(len(columnScore)))
    i=-1
    listIndexInExcel=[]
    for item in columnResponse:
        # print(columnScore[i])
        i=i+1
        strScore = str(columnScore[i])
        strScore.strip()

        # if not (strScore.startswith(prefix) and strScore != strUR):
        #     continue
        listIndexInExcel.append(i)
        # columnResponse[i]=columnResponse[i].replace("<p>", "").replace("</p>", "")
        # strPOS=getPOSFromResponse(columnResponse[i])
        strResponse = str(columnResponse[i]).replace("<p>", "").replace("</p>", "").replace("<br>", "")
        # print(strResponse)

        if strScore == strMF:
            dictScoreResponse[strMF].append(strResponse)
            # dictScoreResponse['I-MF'].append(' ')
            # dictScoreResponse['I-MF'].append(strPOS)
        elif strScore == strMM:
            dictScoreResponse[strMM].append(strResponse)
            # dictScoreResponse['I-MM'].append(' ')
            # dictScoreResponse['I-MM'].append(strPOS)
        elif strScore == strSE:
            dictScoreResponse[strSE].append(strResponse)
            # dictScoreResponse['I-SE'].append(' ')
            # dictScoreResponse['I-SE'].append(strPOS)
        elif strScore == strNE:
            dictScoreResponse[strNE].append(strResponse)
            # dictScoreResponse['I-NE'].append(' ')
            # dictScoreResponse['I-NE'].append(strPOS)

    strTotalIMF = ' '.join(dictScoreResponse[strMF])
    strTotalIMM = ' '.join(dictScoreResponse[strMM])
    strTotalISE = ' '.join(dictScoreResponse[strSE])
    strTotalINE = ' '.join(dictScoreResponse[strNE])

    # print(strTotalIMF)
    corpus = [str(strTotalIMF), str(strTotalIMM), str(strTotalISE), str(strTotalINE)]

    for i in range(len(columnResponse)):
        strScore = str(columnScore[i])
        strScore.strip()
        # if not (strScore.startswith(prefix) and strScore != strUR):
        #     continue
        strResponse = str(columnResponse[i]).replace("<p>", "").replace("</p>", "").replace("<br>", "")
        corpus.append(strResponse)

    vectorizer = TfidfVectorizer(ngram_range=(nGram, nGram))
    X = vectorizer.fit_transform(corpus)
    arrFeatureNames = vectorizer.get_feature_names()
    print('names: ' + str(len(arrFeatureNames)) + ' ' + str(arrFeatureNames))

    dictTopicVectors = {strMF: [], strMM: [], strSE: [], strNE: []}

    dictTopicVectors[strMF] = X[0].todense()
    dictTopicVectors[strMM] = X[1].todense()
    dictTopicVectors[strSE] = X[2].todense()
    dictTopicVectors[strNE] = X[3].todense()

    csv = open(fpOutputYear, 'w')

    columnTitleRow = "no,testId,topic,distIMF,distIMM,distISE,distINE,maxSim,predicted,expected\n"
    csv.write(columnTitleRow)

    print(str(len(corpus)))
    for i in range(4, len(corpus)):
        vectori = X[i].todense()

        # arrIMF = np.array([vectori, dictTopicVectors['I-MF']])
        # arrIMM = np.array([vectori, dictTopicVectors['I-MM']])
        # arrISE = np.array([vectori, dictTopicVectors['I-SE']])
        # arrINE = np.array([vectori, dictTopicVectors['I-NE']])

        # print(" aaaa "+str(pairwise_distances(arrIMF, metric="cosine")))
        # distIMF = pairwise_distances(arrIMF, metric="cosine")[0][1]
        # distIMM = pairwise_distances(arrIMM, metric="cosine")[0][1]
        # distISE = pairwise_distances(arrISE, metric="cosine")[0][1]
        # distINE = pairwise_distances(arrINE, metric="cosine")[0][1]

        distIMF = cosine_similarity(vectori, dictTopicVectors[strMF])[0][0]
        distIMM = cosine_similarity(vectori, dictTopicVectors[strMM])[0][0]
        distISE = cosine_similarity(vectori, dictTopicVectors[strSE])[0][0]
        distINE = cosine_similarity(vectori, dictTopicVectors[strNE])[0][0]

        # distIMF = cosine_similarity(vectori, vectori)[0][0]
        # distIMM = cosine_similarity(vectori, vectori)[0][0]
        # distISE = cosine_similarity(vectori, vectori)[0][0]
        # distINE = cosine_similarity(vectori, vectori)[0][0]

        lst = [distIMF, distIMM, distISE, distINE]
        maxDist = max(lst)

        classResult = strNE
        indexCorpus=listIndexInExcel[i-4]
        expectedResult = columnScore[indexCorpus]
        strTestId=columnTestid[indexCorpus]
        strTopic=columnTopic[indexCorpus]

        if distIMF == maxDist:
            classResult = strMF
        elif distIMM == maxDist:
            classResult = strMM
        elif distISE == maxDist:
            classResult = strSE
        else:
            classResult = strNE

        # print(
        #     str(i) + '\t' + str(distIMF) + '\t' + str(distIMM) + '\t' + str(distISE) + '\t' + str(distINE) + '\t' + str(
        #         maxDist) + '\t' + str(classResult))
        row = str(i - 3)+ ',' + str(strTestId)+ ',' + str(strTopic) + ',' + str(distIMF) + ',' + str(distIMM) + ',' + str(distISE) + ',' + str(
            distINE) + ',' + str(
            maxDist) + ',' + str(classResult) + ',' + str(expectedResult) + '\n'
        csv.write(row)


def main():
    # fpInputYear2018 = 'ratingPositionFilter_2018.csv'
    # fpOutputYear2018 = 'vector_pos_ratingPositionFilter_2018.csv'
    # fpInputYear2019 = 'ratingPositionFilter_2019.csv'
    # fpOutputYear2019 = 'vector_pos_ratingPositionFilter_2019.csv'
    arrNGram=[1,2,3,4]
    fpInput = 'all-formA.csv'

    for idx in range(0,len(arrNGram)):
        folder="10cv_"+str(arrNGram[idx])+"/"

        try:
            # Create target Directory
            os.mkdir(folder)
            print("Directory ", folder, " Created ")
        except FileExistsError:
            print("Directory ", folder, " already exists")

        fpOutputIntermediate = folder+'AI_10cv.csv'
        fpOutputNovice = folder+'AN_10cv.csv'

        fpFormBInput = 'all-formB.csv'
        fpOutputFormBIntermediate = folder+'BI_10cv.csv'
        fpOutputFormBAdvance = folder+'BA_10cv.csv'


        getDataByNGram(fpInput, fpOutputIntermediate,'I-',arrNGram[idx])
        getDataByNGram(fpInput, fpOutputNovice, 'N-',arrNGram[idx])
        getDataByNGram(fpFormBInput, fpOutputFormBIntermediate, 'I-',arrNGram[idx])
        getDataByNGram(fpFormBInput, fpOutputFormBAdvance, 'A-',arrNGram[idx])

main()