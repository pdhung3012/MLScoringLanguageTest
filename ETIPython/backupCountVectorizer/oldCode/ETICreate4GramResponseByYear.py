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

def main():
    my_csv = pd.read_csv('2018.csv')
    columnResponse = my_csv.Response
    # columnPOS = my_csv.pos_tag
    columnScore = my_csv.Score
    dictScoreResponse = {'I-MF': [], 'I-MM': [], 'I-SE': [], 'I-NE': []}

    # nltk.download('punkt')
    # nltk.download('averaged_perceptron_tagger')
    #
    # cess_sents = cess.tagged_sents()
    # Train the unigram tagger
    # uni_tag = ut(cess_sents)

    print(str(len(columnResponse)))
    for i in range(len(columnResponse)):
        # print(column[i])
        strScore = str(columnScore[i])
        strScore.strip()
        # columnResponse[i]=columnResponse[i].replace("<p>", "").replace("</p>", "")
        # strPOS=getPOSFromResponse(columnResponse[i])
        strResponse=str(columnResponse[i]).replace("<p>", "").replace("</p>", "").replace("<br>", "")
        # print(strResponse)

        if strScore == 'I-MF':
            dictScoreResponse['I-MF'].append(strResponse)
            # dictScoreResponse['I-MF'].append(' ')
            # dictScoreResponse['I-MF'].append(strPOS)
        elif strScore == 'I-MM':
            dictScoreResponse['I-MM'].append(strResponse)
            # dictScoreResponse['I-MM'].append(' ')
            # dictScoreResponse['I-MM'].append(strPOS)
        elif strScore == 'I-SE':
            dictScoreResponse['I-SE'].append(strResponse)
            # dictScoreResponse['I-SE'].append(' ')
            # dictScoreResponse['I-SE'].append(strPOS)
        elif strScore == 'I-NE':
            dictScoreResponse['I-NE'].append(strResponse)
            # dictScoreResponse['I-NE'].append(' ')
            # dictScoreResponse['I-NE'].append(strPOS)

    strTotalIMF = ' '.join(dictScoreResponse['I-MF'])
    strTotalIMM = ' '.join(dictScoreResponse['I-MM'])
    strTotalISE = ' '.join(dictScoreResponse['I-SE'])
    strTotalINE = ' '.join(dictScoreResponse['I-NE'])

    # print(strTotalIMF)
    corpus = [str(strTotalIMF), str(strTotalIMM), str(strTotalISE), str(strTotalINE)]

    for i in range(len(columnResponse)):
        strResponse = str(columnResponse[i]).replace("<p>", "").replace("</p>", "").replace("<br>", "")
        corpus.append(strResponse)

    vectorizer = TfidfVectorizer(ngram_range=(1, 4))
    X = vectorizer.fit_transform(corpus)
    arrFeatureNames = vectorizer.get_feature_names()
    print('names: ' + str(len(arrFeatureNames)) + ' ' + str(arrFeatureNames))

    dictTopicVectors = {'I-MF': [], 'I-MM': [], 'I-SE': [], 'I-NE': []}

    dictTopicVectors['I-MF'] = X[0].todense()
    dictTopicVectors['I-MM'] = X[1].todense()
    dictTopicVectors['I-SE'] = X[2].todense()
    dictTopicVectors['I-NE'] = X[3].todense()

    # print(str( dictTopicVectors['I-MF']))

    csv = open('output_Response4Gram_2018.csv', 'w')

    columnTitleRow = "no, distIMF, distIMM, distISE, distINE, maxSim, predicted, expected\n"
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

        distIMF = cosine_similarity(vectori, dictTopicVectors['I-MF'])[0][0]
        distIMM = cosine_similarity(vectori, dictTopicVectors['I-MM'])[0][0]
        distISE = cosine_similarity(vectori, dictTopicVectors['I-SE'])[0][0]
        distINE = cosine_similarity(vectori, dictTopicVectors['I-NE'])[0][0]

        # distIMF = cosine_similarity(vectori, vectori)[0][0]
        # distIMM = cosine_similarity(vectori, vectori)[0][0]
        # distISE = cosine_similarity(vectori, vectori)[0][0]
        # distINE = cosine_similarity(vectori, vectori)[0][0]

        lst = [distIMF, distIMM, distISE, distINE]
        maxDist = max(lst)

        classResult = 'I-NE'
        expectedResult=columnScore[i-4]

        if distIMF == maxDist:
            classResult = 'I-MF'
        elif distIMM == maxDist:
            classResult = 'I-MM'
        elif distISE == maxDist:
            classResult = 'I-SE'
        else:
            classResult = 'I-NE'

        # print(
        #     str(i) + '\t' + str(distIMF) + '\t' + str(distIMM) + '\t' + str(distISE) + '\t' + str(distINE) + '\t' + str(
        #         maxDist) + '\t' + str(classResult))
        row = str(i-3) + ',' + str(distIMF) + ',' + str(distIMM) + ',' + str(distISE) + ',' + str(distINE) + ',' + str(
                maxDist) + ',' + str(classResult)+','+str(expectedResult)+'\n'
        csv.write(row)


main()
