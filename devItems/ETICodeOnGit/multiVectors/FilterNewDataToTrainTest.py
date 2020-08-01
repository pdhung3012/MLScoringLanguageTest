import requests
from multiVectors.utils import createDir,readFile
from datetime import date, timedelta
import os.path
from os import path
import pandas as pd
import csv

fpInput='/home/hung/git/resultETI/combineExcels/combineTest.csv'
fopOutput='/home/hung/git/resultETI/combineExcels/testData/'

createDir(fopOutput)

dfAll=pd.read_csv(fpInput,encoding='latin-1')
dfFilterBA=dfAll[dfAll['Form'].str.startswith('B') & dfAll['Score'].str.startswith('A')  & ~dfAll['Score'].isin(['A-UR'])].groupby(['TestId','Topic','Score'])
dfFilterBI=dfAll[dfAll['Form'].str.startswith('B') & dfAll['Score'].str.startswith('I')  & ~dfAll['Score'].isin(['I-UR'])].groupby(['TestId','Topic','Score'])
dfFilterAN=dfAll[dfAll['Form'].str.startswith('A') & dfAll['Score'].str.startswith('N')  & ~dfAll['Score'].isin(['N-UR'])].groupby(['TestId','Topic','Score'])
dfFilterAI=dfAll[dfAll['Form'].str.startswith('A') & dfAll['Score'].str.startswith('I')  & ~dfAll['Score'].isin(['I-UR'])].groupby(['TestId','Topic','Score'])
# # print(dfFilterBA)
# dfFilterAN.to_csv(fopOutput+'AN.csv')
# dfFilterAI.to_csv(fopOutput+'AI.csv')
# dfFilterBA.to_csv(fopOutput+'BA.csv')
# dfFilterBI.to_csv(fopOutput+'BI.csv')

lstExcels=['TestId,Topic,Score,Response']

for i, g in dfFilterBA:
    # print('abc {}'.format(i))
    # print(g['TestId'].iloc[0] ,' sdsds ',len(g))
    if len(g)==2:
        # print(str(g[0]['TestId']))
        strItem=','.join([str(g['TestId'].iloc[0]),str(g['Topic'].iloc[0]),str(g['Score'].iloc[0]),str(g['Response'].iloc[0])])
        lstExcels.append(strItem)
strAll='\n'.join(lstExcels)
fp=open(fopOutput+'BA.csv','w')
fp.write(strAll)
fp.close()

lstExcels=['TestId,Topic,Score,Response']

for i, g in dfFilterBI:
    # print('abc {}'.format(i))
    # print(g['TestId'].iloc[0] ,' sdsds ',len(g))
    if len(g)==2:
        # print(str(g[0]['TestId']))
        strItem=','.join([str(g['TestId'].iloc[0]),str(g['Topic'].iloc[0]),str(g['Score'].iloc[0]),str(g['Response'].iloc[0])])
        lstExcels.append(strItem)
strAll='\n'.join(lstExcels)
fp=open(fopOutput+'BI.csv','w')
fp.write(strAll)
fp.close()

lstExcels=['TestId,Topic,Score,Response']

for i, g in dfFilterAN:
    # print('abc {}'.format(i))
    # print(g['TestId'].iloc[0] ,' sdsds ',len(g))
    if len(g)==2:
        # print(str(g[0]['TestId']))
        strItem=','.join([str(g['TestId'].iloc[0]),str(g['Topic'].iloc[0]),str(g['Score'].iloc[0]),str(g['Response'].iloc[0])])
        lstExcels.append(strItem)
strAll='\n'.join(lstExcels)
fp=open(fopOutput+'AN.csv','w')
fp.write(strAll)
fp.close()

lstExcels=['TestId,Topic,Score,Response']
for i, g in dfFilterAI:
    # print('abc {}'.format(i))
    # print(g['TestId'].iloc[0] ,' sdsds ',len(g))
    if len(g)==2:
        # print(str(g[0]['TestId']))
        strItem=','.join([str(g['TestId'].iloc[0]),str(g['Topic'].iloc[0]),str(g['Score'].iloc[0]),str(g['Response'].iloc[0])])
        lstExcels.append(strItem)
strAll='\n'.join(lstExcels)
fp=open(fopOutput+'AI.csv','w')
fp.write(strAll)
fp.close()



    # print (g)













