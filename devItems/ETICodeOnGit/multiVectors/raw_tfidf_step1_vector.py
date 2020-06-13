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
from sklearn.decomposition import PCA
import os
from multiVectors.utils import createDir


def getData(fpInputYear,fpOutputYear,prefix):

    my_csv = pd.read_csv(fpInputYear)
    my_csv = my_csv.drop(my_csv[~my_csv.Score.str.startswith(prefix, na=False)].index)
    my_csv = my_csv.drop(my_csv[my_csv.Score.str.endswith('UR')].index)
    my_csv = my_csv.reset_index(drop=True)

    # columnPOS = my_csv.POS
    columnScore = my_csv.Score
    columnTestid = my_csv.Testid
    columnTopic = my_csv.Topic
    columnResponse = my_csv.Response

    corpus = []

    for i in range(len(columnResponse)):
        strScore = str(columnScore[i])
        strScore.strip()
        # if not (strScore.startswith(prefix) and strScore != strUR):
        #     continue
        strResponse = str(columnResponse[i]).replace("<p>", "").replace("</p>", "").replace("<br>", "")
        corpus.append(strResponse)

    vectorizer = TfidfVectorizer(ngram_range=(1, 4))
    X = vectorizer.fit_transform(corpus)
    X = X.toarray()
    # pca = PCA(n_components=50)
    # print('prepare to fit transform')
    # X = pca.fit_transform(X)
    # print('end fit transform')

    arrFeatureNames = vectorizer.get_feature_names()
    print('names: ' + str(len(arrFeatureNames)) + ' ' + str(arrFeatureNames))
    lenVector = len(X[0])

    csv = open(fpOutputYear, 'w')

    columnTitleRow = "no,score,"
    for i in range(0, lenVector):
        item = 'feature-' + str(i + 1)
        columnTitleRow = ''.join([columnTitleRow, item])
        if i != lenVector - 1:
            columnTitleRow = ''.join([columnTitleRow, ","])
    columnTitleRow = ''.join([columnTitleRow, "\n"])
    csv.write(columnTitleRow)
    print(str(len(corpus)))

    for i in range(0, len(corpus)):
        vectori = X[i]
        # print(str(i) + '_' + str(vectori[227]))
        row = ''.join([str(i + 1), ',', str(columnScore[i]), ','])
        strVector = ",".join(str(j) for j in vectori)
        row = ''.join([row, strVector, "\n"])
        # for j in range(0,len(vectori)):
        #     row=''.join([row,',',str(vectori[j])])
        #
        # row = ''.join([row, "\n"])
        csv.write(row)
    csv.close()



def main():
    folder = "../../../../resultETI/raw_tf_idf_4/"
    createDir(folder)

    fpInput = 'all-formA.csv'
    fpOutputIntermediate = folder+'AI_10cv.csv'
    fpOutputNovice = folder+'AN_10cv.csv'

    fpFormBInput = 'all-formB.csv'
    fpOutputFormBIntermediate = folder+'BI_10cv.csv'
    fpOutputFormBAdvance = folder+'BA_10cv.csv'


    getData(fpInput, fpOutputIntermediate,'I-')
    getData(fpInput, fpOutputNovice, 'N-')
    getData(fpFormBInput, fpOutputFormBIntermediate, 'I-')
    getData(fpFormBInput, fpOutputFormBAdvance, 'A-')

main()
