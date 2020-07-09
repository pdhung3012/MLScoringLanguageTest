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
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.tokenize import word_tokenize


def getDataCNN(fpInputYear,fpOutputYear,prefix):

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
    csv = open(fpOutputYear, 'w')
    columnTitleRow = "no,testId,score,response\n"
    csv.write(columnTitleRow)

    for i in range(len(columnResponse)):
        strScore = str(columnScore[i])
        strScore.strip()
        strTestId = str(columnTestid[i])
        strTestId.strip()
        strTopic = str(columnTopic[i])
        strTopic.strip()
        # if not (strScore.startswith(prefix) and strScore != strUR):
        #     continue
        strResponse = str(columnResponse[i]).replace("<p>", "").replace("</p>", "").replace("<br>", "").replace(",", " ")
        strResponse=strResponse.lower()
        corpus.append(strResponse)
        rowAppend='{},{},{},{}\n'.format(str(i+1),strTestId,strScore,strResponse)
        csv.write(rowAppend)
    # tagged_data = [TaggedDocument(words=word_tokenize(_d), tags=[str(i)]) for i, _d in enumerate(corpus)]
    csv.close()



    # max_epochs = 5
    # vec_size = 20
    # alpha = 0.025
    #
    # model = Doc2Vec(size=vec_size,
    #                 alpha=alpha,
    #                 min_alpha=0.00025,
    #                 min_count=1,
    #                 dm=0)
    #
    # model.build_vocab(tagged_data)
    #
    # for epoch in range(max_epochs):
    #     # print('iteration {0}'.format(epoch))
    #     model.train(tagged_data,
    #                 total_examples=model.corpus_count,
    #                 epochs=model.iter)
    #     # decrease the learning rate
    #     model.alpha -= 0.0002
    #     # fix the learning rate, no decay
    #     model.min_alpha = model.alpha
    #     print('End epoch{}'.format(epoch))
    #
    # model.save(fpD2v)
    # # model = Doc2Vec.load(fpModelData)

    # X=[]
    # for item in corpus:
    #     x_data = word_tokenize(item)
    #     v1 = model.infer_vector(x_data)
    #     X.append(v1)


    # vectorizer = TfidfVectorizer(ngram_range=(1, 1))
    # X = vectorizer.fit_transform(corpus)
    # X = X.toarray()
    # pca = PCA(n_components=50)
    # print('prepare to fit transform')
    # X = pca.fit_transform(X)
    # print('end fit transform')

    # arrFeatureNames = vectorizer.get_feature_names()
    # print('names: ' + str(len(arrFeatureNames)) + ' ' + str(arrFeatureNames))
    # lenVector = len(X[0])
    #
    # csv = open(fpOutputYear, 'w')
    #
    # columnTitleRow = "no,score,"
    # for i in range(0, lenVector):
    #     item = 'feature-' + str(i + 1)
    #     columnTitleRow = ''.join([columnTitleRow, item])
    #     if i != lenVector - 1:
    #         columnTitleRow = ''.join([columnTitleRow, ","])
    # columnTitleRow = ''.join([columnTitleRow, "\n"])
    # csv.write(columnTitleRow)
    # print(str(len(corpus)))
    #
    # for i in range(0, len(corpus)):
    #     vectori = X[i]
    #     # print(str(i) + '_' + str(vectori[227]))
    #     row = ''.join([str(i + 1), ',', str(columnScore[i]), ','])
    #     strVector = ",".join(str(j) for j in vectori)
    #     row = ''.join([row, strVector, "\n"])
    #     # for j in range(0,len(vectori)):
    #     #     row=''.join([row,',',str(vectori[j])])
    #     #
    #     # row = ''.join([row, "\n"])
    #     csv.write(row)
    # csv.close()



def main():
    folder = "../../../../resultETI/SpanishRater/inputCsvs/"
    createDir(folder)

    fpInput = 'all-formA.csv'
    fpOutputIntermediate = folder+'AI_text.csv'
    fpOutputNovice = folder+'AN_text.csv'
    # fpOutputNoviceD2v = folder + 'AN_10cv.d2v.txt'
    # fpOutputIntermediateD2v = folder + 'AI_10cv.d2v.txt'

    fpFormBInput = 'all-formB.csv'
    fpOutputFormBIntermediate = folder+'BI_text.csv'
    fpOutputFormBAdvance = folder+'BA_text.csv'
    # fpOutputFormBIntermediateD2v = folder + 'BI_10cv.d2v.txt'
    # fpOutputFormBAdvanceD2v = folder + 'BA_10cv.d2v.txt'


    getDataCNN(fpInput, fpOutputIntermediate,'I-')
    getDataCNN(fpInput, fpOutputNovice, 'N-')
    getDataCNN(fpFormBInput, fpOutputFormBIntermediate, 'I-')
    getDataCNN(fpFormBInput, fpOutputFormBAdvance, 'A-')

main()
