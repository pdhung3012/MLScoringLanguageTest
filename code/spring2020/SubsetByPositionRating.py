import pandas as pd
import os
import re
import csv
from pandas import DataFrame
from nltk.parse import CoreNLPParser
import nltk


def getPOSFromResponse(string,pos_tagger):
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

def checkExactlyAggreeRow(rowEntity):
    rowFilterNA=[]
    for row in rowEntity:
        # print(row['Score'])
        if str(row['Score']) != 'nan':
            rowFilterNA.append(row)

    # print(len(rowFilterNA))
    mapRater1 = {}
    mapRater2 = {}
    for row in rowFilterNA:
        # print('rating ',str(row['Rating position']))
        if str(row['Rating position'])=='1.0':
            mapRater1[row['Topic']]=row
        elif str(row['Rating position'])=='2.0':
            mapRater2[row['Topic']] = row
    resultMatch=True

    rowResult=[]
    rowResultFilter = []
    if len(mapRater1) >0 and  len(mapRater1) == len(mapRater2):
        for key in mapRater1:
            row1 = mapRater1[key]
            row2 = mapRater2[key]

            if key not in mapRater2:
                resultMatch = False
                break
            if str(row1['Score']) != str(row2['Score']):
                resultMatch=False
                break
            else:
                rowResult.append(row1)
                rowResult.append(row2)
                rowResultFilter.append(row1)
    else:
        resultMatch=False

    if resultMatch:
        return rowResult,rowResultFilter
    else:
        return [],[]






def subsetOriginalByPositionRating(strHost,inputFile,outputFile,outputFilterFile,sepStr):
    pw18 = pd.read_csv(inputFile, sep=sepStr,
                       encoding="ISO-8859-1")
    frames = [pw18]
    print(pw18.columns.values)
    # ['Username' 'Gender' 'Grade' 'Grade Level' 'Year of study'
    #  'Type of instruction' 'Relationship to language' 'Rating position'
    #  'Testid' 'Test date' 'Version' 'FinalRating' 'R1' 'Rater1' 'R2' 'Rater2'
    #  'R3' 'Rater3' 'R4' 'Rater4' 'Score' 'Topic' 'Prompt' 'Response'
    #  'Totaltime']

    mapTestId={}
    pos_tagger = CoreNLPParser(url=strHost, tagtype='pos')

    for index, row in pw18.iterrows():
        testId=row["Testid"]
        if testId not in mapTestId:
            mapTestId[testId] =[]
            mapTestId[testId].append(row)
        else:
            mapTestId[testId].append(row)
        # print(index,"\t",row["Testid"],"\t",row["Rating position"],"\t", row["Score"],"\t", row["Topic"],"\t", row["FinalRating"])

    # df = pd.DataFrame(columns = pw18.columns)
    csvFile = open(outputFile, 'w')
    wr = csv.writer(csvFile, quoting=csv.QUOTE_ALL)
    wr.writerow(pw18.columns.values.tolist())

    csvFile2 = open(outputFilterFile, 'w')
    wr2 = csv.writer(csvFile2, quoting=csv.QUOTE_ALL)
    # pw18.columns.append('POS')
    wr2.writerow(pw18.columns.values.tolist())

    print(len(mapTestId))
    for keyTestId in mapTestId:
        valTestId = mapTestId[keyTestId]
        lstRowItem,lstRowItemFilter = checkExactlyAggreeRow(valTestId)
        for row in lstRowItem:
            # print(str(row.values))
            # df.append(row)
            if not str(row['Response']).strip():
            # if str(row['Response'])!='':
                row['POS'] = ''
            else:
                row['POS'] = getPOSFromResponse(str(row['Response']), pos_tagger)
            wr.writerow(row.values.tolist())
        for row in lstRowItemFilter:
            # print(str(row.values))
            # df.append(row)
            if not str(row['Response']).strip():
            # if str(row['Response'])!='':
                row['POS'] = ''
            else:
                row['POS'] = getPOSFromResponse(str(row['Response']), pos_tagger)

            # print(row['POS'])
            wr2.writerow(row.values.tolist())
        # print('rowItem ',len(lstRowItem))
        # if len(lstRowItem) > 0:
        #     break

def subsetOriginalByPositionRatingChangeData(strHost,inputFile,outputFile,outputFilterFile,sepStr):
    pw18 = pd.read_csv(inputFile, sep=sepStr,
                       encoding="ISO-8859-1")
    frames = [pw18]
    print(pw18.columns.values)
    # ['Username' 'Gender' 'Grade' 'Grade Level' 'Year of study'
    #  'Type of instruction' 'Relationship to language' 'Rating position'
    #  'Testid' 'Test date' 'Version' 'FinalRating' 'R1' 'Rater1' 'R2' 'Rater2'
    #  'R3' 'Rater3' 'R4' 'Rater4' 'Score' 'Topic' 'Prompt' 'Response'
    #  'Totaltime']

    mapTestId={}
    pos_tagger = CoreNLPParser(url=strHost, tagtype='pos')

    for index, row in pw18.iterrows():
        testId=row["Testid"]
        if testId not in mapTestId:
            mapTestId[testId] =[]
            mapTestId[testId].append(row)
        else:
            mapTestId[testId].append(row)
        # print(index,"\t",row["Testid"],"\t",row["Rating position"],"\t", row["Score"],"\t", row["Topic"],"\t", row["FinalRating"])

    # df = pd.DataFrame(columns = pw18.columns)
    csvFile = open(outputFile, 'w')
    wr = csv.writer(csvFile, quoting=csv.QUOTE_ALL)
    wr.writerow(pw18.columns.values.tolist())

    csvFile2 = open(outputFilterFile, 'w')
    wr2 = csv.writer(csvFile2, quoting=csv.QUOTE_ALL)
    # pw18.columns.append('POS')
    wr2.writerow(pw18.columns.values.tolist())

    print(len(mapTestId))
    for keyTestId in mapTestId:
        valTestId = mapTestId[keyTestId]
        lstRowItem,lstRowItemFilter = checkExactlyAggreeRow(valTestId)
        for row in lstRowItem:
            # print(str(row.values))
            # df.append(row)
            if not str(row['Response']).strip():
            # if str(row['Response'])!='':
                row['POS'] = ''
            else:
                row['POS'] = getPOSFromResponse(str(row['Response']), pos_tagger)
            wr.writerow(row.values.tolist())
        for row in lstRowItemFilter:
            # print(str(row.values))
            # df.append(row)
            if not str(row['Response']).strip():
            # if str(row['Response'])!='':
                row['POS'] = ''
            else:
                row['POS'] = getPOSFromResponse(str(row['Response']), pos_tagger)

            # print(row['POS'])
            wr2.writerow(row.values.tolist())
        # print('rowItem ',len(lstRowItem))
        # if len(lstRowItem) > 0:
        #     break



fpOrigin2018 = 'AAPPLResults_071119_2018Schema.csv'
fpRatingPostition2018 = 'ratingPosition_2018.csv'
fpRatingPostitionFilter2018 = 'ratingPositionFilter_2018.csv'
fpOrigin2019 = 'AAPPLResults_071119_2019Schema.csv'
fpRatingPostition2019 = 'ratingPosition_2019.csv'
fpRatingPostitionFilter2019 = 'ratingPositionFilter_2019.csv'

fpOriginNew2019 = 'AAPPLResults_102419.csv'
fpRatingPostitionNew2019 = 'ratingPosition_newData_2019.csv'
fpRatingPostitionFilterNew2019 = 'ratingPositionFilter_newData_2019.csv'


urlHost='http://localhost:9001'

# subsetOriginalByPositionRating(urlHost,fpOrigin2018,fpRatingPostition2018,fpRatingPostitionFilter2018,';')
# subsetOriginalByPositionRating(urlHost,fpOrigin2019,fpRatingPostition2019,fpRatingPostitionFilter2019,';')
subsetOriginalByPositionRatingChangeData(urlHost,fpOriginNew2019,fpRatingPostitionNew2019,fpRatingPostitionFilterNew2019,',')