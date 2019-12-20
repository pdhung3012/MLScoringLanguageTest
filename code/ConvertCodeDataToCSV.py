import pandas as pd
import os
import re
import csv
from pandas import DataFrame



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





def convertToSemiColonSeparate(inputFile, outputFile):
    pw18 = pd.read_csv(inputFile, sep='$',
                       encoding="UTF-8")
    frames = [pw18]
    print(pw18.columns.values)
    # ['Username' 'Gender' 'Grade' 'Grade Level' 'Year of study'
    #  'Type of instruction' 'Relationship to language' 'Rating position'
    #  'Testid' 'Test date' 'Version' 'FinalRating' 'R1' 'Rater1' 'R2' 'Rater2'
    #  'R3' 'Rater3' 'R4' 'Rater4' 'Score' 'Topic' 'Prompt' 'Response'
    #  'Totaltime']

    mapTestId={}

    # for index, row in pw18.iterrows():
    #     testId=row["Testid"]
    #     if testId not in mapTestId:
    #         mapTestId[testId] =[]
    #         mapTestId[testId].append(row)
    #     else:
    #         mapTestId[testId].append(row)
    #     # print(index,"\t",row["Testid"],"\t",row["Rating position"],"\t", row["Score"],"\t", row["Topic"],"\t", row["FinalRating"])

    # df = pd.DataFrame(columns = pw18.columns)
    csvFile = open(outputFile, 'w')
    wr = csv.writer(csvFile, quoting=csv.QUOTE_ALL)
    wr.writerow(pw18.columns.values.tolist())

    row_count = sum(1 for row in pw18)
    print(row_count)
    for index, row in pw18.iterrows():
        wr.writerow(row.values.tolist())
    # print(len(mapTestId))
    # for keyTestId in mapTestId:
    #     valTestId = mapTestId[keyTestId]
    #     lstRowItem,lstRowItemFilter = checkExactlyAggreeRow(valTestId)
    #     for row in lstRowItem:
    #         # print(str(row.values))
    #         # df.append(row)
    #         wr.writerow(row.values.tolist())
    #     for row in lstRowItemFilter:
    #         # print(str(row.values))
    #         # df.append(row)
    #         wr2.writerow(row.values.tolist())
    #     # print('rowItem ',len(lstRowItem))
    #     # if len(lstRowItem) > 0:
    #     #     break

fpInputData = '/Users/hungphan/git/FixUndeclaredVariableProject/prutodb/prutor-deepfix-09-12-2017/errorProgram.csv'
fpConvertData = '/Users/hungphan/git/FixUndeclaredVariableProject/prutodb/prutor-deepfix-09-12-2017/convertProgram.csv'

convertToSemiColonSeparate(fpInputData, fpConvertData)