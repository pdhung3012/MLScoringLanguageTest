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
from sklearn.model_selection import cross_val_score,cross_val_predict, KFold, StratifiedKFold

import os;
from nltk.parse import CoreNLPParser

def getPOSFromResponsePerLine(string,pos_tagger):
    # li = list(string.split(" "))
    string = str(string).replace("<p>", "").replace("</p>", "")
    tokens = nltk.word_tokenize(string)
    # str2 = uni_tag.tag(tokens)
    if not string.strip():
        # print('empty')
        return ''
    arr = pos_tagger.tag(tokens)
    str2=""
    for(pos,tag) in arr:
        str2 = str2 + tag + " "
    newStr = str(str2).replace("[", "").replace("]", "").replace("'", "").replace(",", " ").strip()
    return newStr

def getLemmFromCombineResponse(string,lem_tagger):
    # li = list(string.split(" "))
    string = str(string).replace("<p>", "").replace("</p>", "")
    tokens = nltk.word_tokenize(string)
    # str2 = uni_tag.tag(tokens)
    if not string.strip():
        # print('empty')
        return ''
    arr = lem_tagger.tokenize(string)
    str2=""
    # print(arr[0])
    for lem in arr:
        str2 = str2 + lem + " "
    newStr = str(str2).replace("[", "").replace("]", "").replace("'", "").replace(",", " ").strip()
    return newStr

def getColumnPOS(columnResponse,pos_tagger):
    lstPOS=[]

    for i in range(0,len(columnResponse)):
        strResponse = str(columnResponse[i]).replace("<p>", "").replace("</p>", "").replace("<br>", "")
        strPOS=getPOSFromResponsePerLine(strResponse,pos_tagger)
        lstPOS.append(strPOS)

    return lstPOS

def getColumnLem(columnResponse,lem_tagger):
    lstResponse=[]

    for i in range(0,len(columnResponse)):
        strResponse = str(columnResponse[i]).replace("<p>", "").replace("</p>", "").replace("<br>", "")
        strLem=getLemmFromCombineResponse(strResponse,lem_tagger)
        lstResponse.append(strLem)

    return lstResponse


def createDir(folderName):
    try:
        # Create target Directory
        os.mkdir(folderName)
        print("Directory ", folderName, " Created ")
    except FileExistsError:
        print("Directory ", folderName, " already exists")

def concatStringExcludeIndexes(arr,beginIndex,endIndex):
    lstInput=[]
    for index in range(0,len(arr)):
        if(index<beginIndex or index > endIndex):
            lstInput.append(arr[index])
    return ' '.join(lstInput)

def concatStringExcludeIndexesMultiArr(arrA,arrB,arrC,arrD,trainIndexList):
    # lstInput=[]
    lstTotal=[]
    strA=''
    strB=''
    strC=''
    strD=''
    for index in range(0,len(arrA)):
        if index in trainIndexList:
            # lstInput.append(arrA[index])
            strA=' '.join([strA,arrA[index],'\n'])
            strB = ' '.join([strB, arrB[index],'\n'])
            strC = ' '.join([strC, arrC[index],'\n'])
            strD = ' '.join([strD, arrD[index],'\n'])
        # else:
        #     print(str(index)+' here in test')
    lstTotal.append(strA)
    lstTotal.append(strB)
    lstTotal.append(strC)
    lstTotal.append(strD)

    return lstTotal

def getData(pos_tagger,lem_tagger,fpInputYear,folderName, formElementName,prefix):
    my_csv = pd.read_csv(fpInputYear)
    # filtered = my_csv.Score.str.match("I-",na=False)
    # print(str(len(my_csv)))
    # https://kite.com/python/answers/how-to-delete-rows-from-a-pandas-%60dataframe%60-based-on-a-conditional-expression-in-python
    my_csv = my_csv.drop(my_csv[~my_csv.Score.str.startswith(prefix)].index)
    my_csv = my_csv.drop(my_csv[my_csv.Score.str.endswith('UR')].index)
    # my_csv = my_csv.query("LEFT(Score)")
    my_csv=my_csv.reset_index(drop=True)

    fopTextFolder=folderName+"text/"
    createDir(fopTextFolder)

    # print(my_csv)
    # print(str(len(my_csv)))

    columnScore = my_csv.Score
    columnTestid = my_csv.Testid
    columnTopic = my_csv.Topic
    columnResponse = getColumnLem(my_csv.Response,lem_tagger)
    columnPOS = getColumnPOS(columnResponse,pos_tagger)


    # print(str(len(columnScore)))
    # print('aaaa\t'+str(columnScore[1]))
    strMF = prefix+'MF'
    strMM = prefix+'MM'
    strSE = prefix+'SE'
    strNE = prefix+'NE'
    strUR = prefix+'UR'
    # numFoldTotal = 10
    # lenRowTotal=len(columnResponse)
    # rangeFold= (lenRowTotal//numFoldTotal)
    # print(str(lenRowTotal)+"\t"+str(rangeFold))

    foldIndex=0


    dictScorePOS = {strMF: [], strMM: [], strSE: [], strNE: []}
    # training data
    for i in range(0, len(columnResponse)):
        strScore = str(columnScore[i])
        strScore.strip()

        # if (i >= beginIndex and i <= endIndex):
        #     continue

        # if not (strScore.startswith(prefix) and strScore != strUR):
        #     continue
        # listIndexInExcel.append(i)
        strPOS = str(columnPOS[i])+' '+str(columnResponse[i])

        if strScore == strMF:
            dictScorePOS[strMF].append(strPOS)
            dictScorePOS[strMM].append("")
            dictScorePOS[strSE].append("")
            dictScorePOS[strNE].append("")
        elif strScore == strMM:
            dictScorePOS[strMM].append(strPOS)
            dictScorePOS[strMF].append("")
            dictScorePOS[strSE].append("")
            dictScorePOS[strNE].append("")
        elif strScore == strSE:
            dictScorePOS[strSE].append(strPOS)
            dictScorePOS[strMM].append("")
            dictScorePOS[strMF].append("")
            dictScorePOS[strNE].append("")
        elif strScore == strNE:
            dictScorePOS[strNE].append(strPOS)
            dictScorePOS[strMM].append("")
            dictScorePOS[strSE].append("")
            dictScorePOS[strMF].append("")

    kf = StratifiedKFold(n_splits=10, random_state=10, shuffle=True)

    # if foldIndex == 1:
    fpTestFold = folderName + formElementName + '-all.csv'
    csv = open(fpTestFold, 'w')
    columnTitleRow = "no,testId,topic,distIMF,distIMM,distISE,distINE,expected,pos,response\n"
    csv.write(columnTitleRow)

    for train_index, test_index in kf.split(my_csv,columnScore):
        # print("TRAIN:", train_index, "TEST:", test_index)
        # my_csv_train = my_csv.iloc(train_index)
        # my_csv_test = my_csv.iloc(test_index)

        foldIndex=foldIndex+1

        listIndexInExcel = []
        # beginIndex = (foldIndex - 1) * rangeFold
        # endIndex = foldIndex * rangeFold - 1
        # if foldIndex == numFoldTotal:
        #     endIndex = lenRowTotal - 1

        lstTotal = concatStringExcludeIndexesMultiArr(dictScorePOS[strMF], dictScorePOS[strMM],
                                                      dictScorePOS[strSE], dictScorePOS[strNE], train_index)
        # strTotalIMF = concatStringExcludeIndexes(dictScorePOS[strMF],beginIndex,endIndex)
        # strTotalIMM = concatStringExcludeIndexes(dictScorePOS[strMM],beginIndex,endIndex)
        # strTotalISE = concatStringExcludeIndexes(dictScorePOS[strSE],beginIndex,endIndex)
        # strTotalINE = concatStringExcludeIndexes(dictScorePOS[strNE],beginIndex,endIndex)

        strTotalIMF = lstTotal[0]
        strTotalIMM = lstTotal[1]
        strTotalISE = lstTotal[2]
        strTotalINE = lstTotal[3]

        # print(strTotalIMF)
        corpusTrain = [str(strTotalIMF), str(strTotalIMM), str(strTotalISE), str(strTotalINE)]
        corpusTest = []
        # print("I-MF "+str(strTotalIMF))
        fpTestTextContent = ''.join([fopTextFolder, formElementName, '_test_', str(foldIndex), '_lines.txt'])
        fTestText = open(fpTestTextContent, 'w')

        for i in test_index:
            # strScore = str(columnScore[i])
            # strScore.strip()
            # if not (strScore.startswith(prefix) and strScore != strUR):
            #     continue
            strPOS = str(columnPOS[i])+' '+str(columnResponse[i])
            fTestText.write(strPOS+'\n')
            corpusTest.append(strPOS)
            listIndexInExcel.append(i)
        fTestText.close()

        vectorizer = TfidfVectorizer(ngram_range=(1, 4))
        X = vectorizer.fit_transform(corpusTrain)
        Y = vectorizer.transform(corpusTest)
        arrFeatureNames = vectorizer.get_feature_names()
        # print('names: ' + str(len(arrFeatureNames)) + ' ' + str(arrFeatureNames))

        dictTopicVectors = {strMF: [], strMM: [], strSE: [], strNE: []}

        dictTopicVectors[strMF] = X[0].todense()
        dictTopicVectors[strMM] = X[1].todense()
        dictTopicVectors[strSE] = X[2].todense()
        dictTopicVectors[strNE] = X[3].todense()


        fpTrainTextContentMF=''.join([fopTextFolder,formElementName,'_train_',str(foldIndex),'_'+strMF,'.txt'])
        fTrainTextMF=open(fpTrainTextContentMF,'w')
        fTrainTextMF.write(strTotalIMF)
        fTrainTextMF.close()
        fpTrainTextContentMM = ''.join([fopTextFolder, formElementName, '_train_', str(foldIndex), '_' + strMM, '.txt'])
        fTrainTextMM = open(fpTrainTextContentMM, 'w')
        fTrainTextMM.write(strTotalIMM)
        fTrainTextMM.close()
        fpTrainTextContentSE = ''.join([fopTextFolder, formElementName, '_train_', str(foldIndex), '_' + strSE, '.txt'])
        fTrainTextSE = open(fpTrainTextContentSE, 'w')
        fTrainTextSE.write(strTotalISE)
        fTrainTextSE.close()
        fpTrainTextContentNE = ''.join([fopTextFolder, formElementName, '_train_', str(foldIndex), '_' + strNE, '.txt'])
        fTrainTextNE = open(fpTrainTextContentNE, 'w')
        fTrainTextNE.write(strTotalINE)
        fTrainTextNE.close()



        for i in range(0, len(corpusTest)):
            vectori = Y[i].todense()
            distIMF = cosine_similarity(vectori, dictTopicVectors[strMF])[0][0]
            distIMM = cosine_similarity(vectori, dictTopicVectors[strMM])[0][0]
            distISE = cosine_similarity(vectori, dictTopicVectors[strSE])[0][0]
            distINE = cosine_similarity(vectori, dictTopicVectors[strNE])[0][0]

            lst = [distIMF, distIMM, distISE, distINE]
            maxDist = max(lst)

            classResult = strNE
            indexCorpus = listIndexInExcel[i]
            expectedResult = columnScore[indexCorpus]
            strTestId = columnTestid[indexCorpus]
            strTopic = columnTopic[indexCorpus]
            strPOS=columnPOS[indexCorpus]
            strResponse = str(columnResponse[indexCorpus]).replace(",", "COMA")
            #
            # if distIMF == maxDist:
            #     classResult = strMF
            # elif distIMM == maxDist:
            #     classResult = strMM
            # elif distISE == maxDist:
            #     classResult = strSE
            # else:
            #     classResult = strNE

            row = ''.join([str(indexCorpus + 1), ',' , str(strTestId), ',' , str(strTopic), ',' + str(distIMF), ',', str(
                distIMM), ',', str(distISE) + ',', str(distINE), ',', str(expectedResult)+ ',', str(strPOS), ',', str(strResponse), '\n'])
            csv.write(row)
        print("End fold" + str(foldIndex) + " " + str(len(corpusTest)))
    csv.close()
    print("End level ")
    # print(fpOutputTest)


def main():
    # fpInputYear2018 = 'ratingPositionFilter_2018.csv'
    # fpOutputYear2018 = 'vector_pos_ratingPositionFilter_2018.csv'
    # fpInputYear2019 = 'ratingPositionFilter_2019.csv'
    # fpOutputYear2019 = 'vector_pos_ratingPositionFilter_2019.csv'

    folderName = '10cvPOSResponse/'
    try:
        # Create target Directory
        os.mkdir(folderName)
        print("Directory ", folderName, " Created ")
    except FileExistsError:
        print("Directory ", folderName, " already exists")

    fpInput = 'all-formA.csv'
    # fpOutputIntermediate = folderName+ 'AI-all.csv'
    # fpOutputNovice = folderName+'AN-all.csv'

    fpFormBInput = 'all-formB.csv'
    # fpOutputFormBIntermediate = folderName+'BI-all.csv'
    # fpOutputFormBAdvance = folderName+'BA-all.csv'
    urlHost = 'http://localhost:9001'
    pos_tagger = CoreNLPParser(url=urlHost, tagtype='pos')
    lem_tagger= CoreNLPParser(url=urlHost, tagtype='pos')
    # getData(fpInputYear2018,fpOutputYear2018)
    # getData(fpInputYear2019, fpOutputYear2019)
    getData(pos_tagger,lem_tagger,fpInput,folderName, 'AI','I-')
    getData(pos_tagger,lem_tagger,fpInput,folderName, 'AN', 'N-')
    getData(pos_tagger,lem_tagger,fpFormBInput,folderName, 'BI', 'I-')
    getData(pos_tagger,lem_tagger,fpFormBInput,folderName, 'BA', 'A-')

main()
