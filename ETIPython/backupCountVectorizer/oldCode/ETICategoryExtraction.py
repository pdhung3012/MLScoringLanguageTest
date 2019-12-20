from sklearn.feature_extraction.text import TfidfVectorizer

import pandas as pd
from scipy import spatial
import numpy as np
from sklearn.metrics import pairwise_distances
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import cosine


def main():
    my_csv = pd.read_csv('scorepoints_only4.csv')
    columnResponse = my_csv.Response
    columnScore = my_csv.Score
    dictScoreResponse = {'I-MF': [], 'I-MM': [], 'I-SE': [], 'I-NE': []}

    for i in range(len(columnResponse)):
        # print(column[i])
        strScore = str(columnScore[i])
        strScore.strip()
        # print(strScore == 'I-MF')
        if strScore == 'I-MF':
            dictScoreResponse['I-MF'].append(columnResponse[i])
        elif strScore == 'I-MM':
            dictScoreResponse['I-MM'].append(columnResponse[i])
        elif strScore == 'I-SE':
            dictScoreResponse['I-SE'].append(columnResponse[i])
        elif strScore == 'I-NE':
            dictScoreResponse['I-NE'].append(columnResponse[i])

    strTotalIMF = ' '.join(dictScoreResponse['I-MF'])
    strTotalIMM = ' '.join(dictScoreResponse['I-MM'])
    strTotalISE = ' '.join(dictScoreResponse['I-SE'])
    strTotalINE = ' '.join(dictScoreResponse['I-NE'])

    # print(strTotalIMF)
    corpus = [str(strTotalIMF), str(strTotalIMM), str(strTotalISE), str(strTotalINE)]

    for i in range(len(columnResponse)):
        corpus.append(columnResponse[i])

    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(corpus)
    arrFeatureNames = vectorizer.get_feature_names()
    print('names: ' + str(len(arrFeatureNames)) + ' ' + str(arrFeatureNames))

    dictTopicVectors = {'I-MF': [], 'I-MM': [], 'I-SE': [], 'I-NE': []}

    dictTopicVectors['I-MF'] = X[0].todense()
    dictTopicVectors['I-MM'] = X[1].todense()
    dictTopicVectors['I-SE'] = X[2].todense()
    dictTopicVectors['I-NE'] = X[3].todense()

    # print(str( dictTopicVectors['I-MF']))

    csv = open('output_similarity.csv', 'w')

    columnTitleRow = "no, distIMF, distIMM, distISE, distINE, maxSim, predicted, expected\n"
    csv.write(columnTitleRow)

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
