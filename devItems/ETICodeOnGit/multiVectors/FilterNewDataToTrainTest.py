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
dfFilterBA=dfAll[dfAll['Form'].str.startswith('B') & dfAll['Score'].str.startswith('A')  & ~dfAll['Score'].isin(['A-UR'])]
dfFilterBI=dfAll[dfAll['Form'].str.startswith('B') & dfAll['Score'].str.startswith('I')  & ~dfAll['Score'].isin(['I-UR'])]
dfFilterAN=dfAll[dfAll['Form'].str.startswith('A') & dfAll['Score'].str.startswith('N')  & ~dfAll['Score'].isin(['N-UR'])]
dfFilterAI=dfAll[dfAll['Form'].str.startswith('A') & dfAll['Score'].str.startswith('I')  & ~dfAll['Score'].isin(['I-UR'])]
# print(dfFilterBA)
dfFilterAN.to_csv(fopOutput+'AN.csv')
dfFilterAI.to_csv(fopOutput+'AI.csv')
dfFilterBA.to_csv(fopOutput+'BA.csv')
dfFilterBI.to_csv(fopOutput+'BI.csv')















