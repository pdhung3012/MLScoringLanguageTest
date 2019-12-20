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
    my_csv = pd.read_csv('advance_2019.csv')
    columnResponse = my_csv.Response
    # columnPOS = my_csv.pos_tag
    columnScore = my_csv.Score
    dictScoreResponse = {'A-MF': [], 'A-MM': [], 'A-SE': [], 'A-NE': []}

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

        if strScore == 'A-MF':
            dictScoreResponse['A-MF'].append(strResponse)
            # dictScoreResponse['A-MF'].append(' ')
            # dictScoreResponse['A-MF'].append(strPOS)
        elif strScore == 'A-MM':
            dictScoreResponse['A-MM'].append(strResponse)
            # dictScoreResponse['A-MM'].append(' ')
            # dictScoreResponse['A-MM'].append(strPOS)
        elif strScore == 'A-SE':
            dictScoreResponse['A-SE'].append(strResponse)
            # dictScoreResponse['A-SE'].append(' ')
            # dictScoreResponse['A-SE'].append(strPOS)
        elif strScore == 'A-NE':
            dictScoreResponse['A-NE'].append(strResponse)
            # dictScoreResponse['A-NE'].append(' ')
            # dictScoreResponse['A-NE'].append(strPOS)

    strTotalIMF = ' '.join(dictScoreResponse['A-MF'])
    strTotalIMM = ' '.join(dictScoreResponse['A-MM'])
    strTotalISE = ' '.join(dictScoreResponse['A-SE'])
    strTotalINE = ' '.join(dictScoreResponse['A-NE'])

    # print(strTotalIMF)
    corpus = [str(strTotalIMF), str(strTotalIMM), str(strTotalISE), str(strTotalINE)]

    for i in range(len(columnResponse)):
        strResponse = str(columnResponse[i]).replace("<p>", "").replace("</p>", "").replace("<br>", "")
        corpus.append(strResponse)

    vectorizer = TfidfVectorizer(ngram_range=(1, 4))
    X = vectorizer.fit_transform(corpus)
    arrFeatureNames = vectorizer.get_feature_names()
    print('names: ' + str(len(arrFeatureNames)) + ' ' + str(arrFeatureNames))

    dictTopicVectors = {'A-MF': [], 'A-MM': [], 'A-SE': [], 'A-NE': []}

    dictTopicVectors['A-MF'] = X[0].todense()
    dictTopicVectors['A-MM'] = X[1].todense()
    dictTopicVectors['A-SE'] = X[2].todense()
    dictTopicVectors['A-NE'] = X[3].todense()

    # print(str( dictTopicVectors['A-MF']))

    csv = open('advance_outRes4gram_2019.csv', 'w')

    columnTitleRow = "no, distIMF, distIMM, distISE, distINE, maxSim, predicted, expected\n"
    csv.write(columnTitleRow)

    print(str(len(corpus)))
    for i in range(4, len(corpus)):
        vectori = X[i].todense()

        # arrIMF = np.array([vectori, dictTopicVectors['A-MF']])
        # arrIMM = np.array([vectori, dictTopicVectors['A-MM']])
        # arrISE = np.array([vectori, dictTopicVectors['A-SE']])
        # arrINE = np.array([vectori, dictTopicVectors['A-NE']])

        # print(" aaaa "+str(pairwise_distances(arrIMF, metric="cosine")))
        # distIMF = pairwise_distances(arrIMF, metric="cosine")[0][1]
        # distIMM = pairwise_distances(arrIMM, metric="cosine")[0][1]
        # distISE = pairwise_distances(arrISE, metric="cosine")[0][1]
        # distINE = pairwise_distances(arrINE, metric="cosine")[0][1]

        distIMF = cosine_similarity(vectori, dictTopicVectors['A-MF'])[0][0]
        distIMM = cosine_similarity(vectori, dictTopicVectors['A-MM'])[0][0]
        distISE = cosine_similarity(vectori, dictTopicVectors['A-SE'])[0][0]
        distINE = cosine_similarity(vectori, dictTopicVectors['A-NE'])[0][0]

        # distIMF = cosine_similarity(vectori, vectori)[0][0]
        # distIMM = cosine_similarity(vectori, vectori)[0][0]
        # distISE = cosine_similarity(vectori, vectori)[0][0]
        # distINE = cosine_similarity(vectori, vectori)[0][0]

        lst = [distIMF, distIMM, distISE, distINE]
        maxDist = max(lst)

        classResult = 'A-NE'
        expectedResult=columnScore[i-4]

        if distIMF == maxDist:
            classResult = 'A-MF'
        elif distIMM == maxDist:
            classResult = 'A-MM'
        elif distISE == maxDist:
            classResult = 'A-SE'
        else:
            classResult = 'A-NE'

        # print(
        #     str(i) + '\t' + str(distIMF) + '\t' + str(distIMM) + '\t' + str(distISE) + '\t' + str(distINE) + '\t' + str(
        #         maxDist) + '\t' + str(classResult))
        row = str(i-3) + ',' + str(distIMF) + ',' + str(distIMM) + ',' + str(distISE) + ',' + str(distINE) + ',' + str(
                maxDist) + ',' + str(classResult)+','+str(expectedResult)+'\n'
        csv.write(row)


main()
