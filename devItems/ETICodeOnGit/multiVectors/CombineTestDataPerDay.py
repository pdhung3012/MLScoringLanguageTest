import requests
from multiVectors.utils import createDir,readFile
from datetime import date, timedelta
import os.path
from os import path
import pandas as pd
import csv

folderInputDays = "../../../../resultETI/dataCollectionFromMarch1st/"

createDir(folderInputDays)

folderCombineFiles="../../../../resultETI/combineExcels/"
createDir(folderCombineFiles)


fpLogDays= folderInputDays + 'logDays.txt'
fpCombineFiles=folderCombineFiles+'combineTest.csv'

strHeader2='TestId,Ratingposition,TestVersion,FinalRating,Score,Rater1_Id,Rater2_Id,Rater3_Id,Rater4_Id,UserName'
strHeader1='Topic,Response,TestVersion,TestId,TotalTime,UserName'


sdate = date(2020, 3, 1)   # start date
edate = date.today()   # end date

delta = edate - sdate       # as timedelta
lstDays=[]
indx=0
lstDfs=[]
lstTotalLine = ['No,TestId,Form,Rater1,Rater2,Rater3,Rater4,Topic,Score,Response']

for i in range(delta.days + 1):
    day = sdate + timedelta(days=i)
    strDay=str(day)
    fpDayT1=folderInputDays+strDay+'_1.csv'
    fpDayT2 = folderInputDays + strDay + '_2.csv'
    fpDayT1Revise = folderInputDays + strDay + '_1_revise.csv'
    fpDayT2Revise = folderInputDays + strDay + '_2_revise.csv'



    if not path.exists(fpDayT1):
        continue

    import codecs

    f1 = codecs.open(fpDayT1, 'r', encoding='latin-1')
    # f1=open(fpDayT1,'r', encoding="utf8")
    strContent1 = f1.read()
    f1.close()
    arrContent1 = strContent1.split('\n')
    f1 = codecs.open(fpDayT2, 'r', encoding='latin-1')
    # f1=open(fpDayT1,'r', encoding="utf8")
    strContent2 = f1.read()
    f1.close()
    arrContent2 = strContent2.split('\n')

    isInQuote = False
    lstRevise = []
    lstT2Revise=[]

    for iii in range(len(arrContent1)):
        strOldItem = arrContent1[iii].strip()
        if strOldItem == "":
            continue
        l = list(strOldItem)
        lstStr = []

        # print(strOldItem)
        rquote=strOldItem.rindex("\"")
        lquote=strOldItem.index("\"")

        # print('{}'.format(arrContent2[iii]))

        for id2 in range(len(l)):
            charItem = l[id2]
            if id2>=lquote and id2<=rquote:
                if not charItem in [',',';','"']:
                    lstStr.append(charItem)
            else:
                lstStr.append(charItem)

        strNewItem = ''.join(lstStr)
        strNewItem=strNewItem.replace("<p>", "").replace("</p>", "").replace("<br>", "")
        lstRevise.append(strNewItem)
        strT2Item=arrContent2[iii]
        lstT2Revise.append(strT2Item)

    # print(str(lstRevise[0]))
    # if str(lstRevise[0]) != strHeader1:
    #     lstRevise.insert(0,strHeader1)
    #     print(lstRevise)
    strRevise = '\n'.join(lstRevise)
    # strRevise=strRevise.replace("<p>", "").replace("</p>", "").replace("<br>", "")
    fp1 = open(fpDayT1Revise, 'w')
    fp1.write(strRevise)
    fp1.close()

    # if str(lstT2Revise[0]) != strHeader2:
    #     lstT2Revise.insert(0, strHeader2)

    strRevise = '\n'.join(lstT2Revise)
    # strRevise=strRevise.replace("<p>", "").replace("</p>", "").replace("<br>", "")
    fp1 = open(fpDayT2Revise, 'w')
    fp1.write(strRevise)
    fp1.close()



    # df1=pd.read_csv(fpDayT1Revise, engine="python", sep=',', quotechar='"', error_bad_lines=False)
    # df2 = pd.read_csv(fpDayT2, engine="python", sep=',', quotechar='"', error_bad_lines=False)
    df1 = pd.read_csv(fpDayT1Revise)
    df2 = pd.read_csv(fpDayT2Revise)

    if(len(df1)!=len(df2)):
        continue
    indx = indx + 1
    print(fpDayT1)
    print('{}\t{}\t{}'.format(indx,len(df1),len(df2)))

    try:
        for iii in range(len(df1['Response'])):
            strLineCombine=','.join([str((iii+1)),str(df1['TestId'][iii]),str(df1['TestVersion'][iii]),str(df2['Rater1_Id'][iii]),str(df2['Rater2_Id'][iii]),str(df2['Rater3_Id'][iii]),str(df2['Rater4_Id'][iii]),str(df1['Topic'][iii]),str(df2['Score'][iii]),str(df1['Response'][iii])])
            lstTotalLine.append(strLineCombine)
    except:
        print('error')

strRevise = '\n'.join(lstTotalLine)
# strRevise=strRevise.replace("<p>", "").replace("</p>", "").replace("<br>", "")
fp1 = open(fpCombineFiles, 'w')
fp1.write(strRevise)
fp1.close()

    # frames = [df1, df2]
    # df3 = pd.concat(frames)
    # lstDfs.append(df3)

# dfAll=pd.concat(lstDfs)
# dfAll.to_csv(fpCombineFiles, index=False)




