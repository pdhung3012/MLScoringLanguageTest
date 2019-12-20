import pandas as pd
import os
import re

pw18 = pd.read_csv('/Users/hungphan/Downloads/AAPPLResults_071119_2018Schema.csv', sep=';',
                   encoding="ISO-8859-1")
pw19 = pd.read_csv('/Users/hungphan/Downloads/AAPPLResults_071119_2019Schema.csv', sep=';',
                   encoding="ISO-8859-1")
# frames = [pw18, pw19]
frames = [pw18]
result = pd.concat(frames)
agreed = result[result['R1'] == result['R2']]
# agreed['Score'].value_counts().plot(kind='bar')
# agreed['Score'].value_counts()
# agreed['FinalRating'].value_counts().plot(kind='bar')
# agreed['FinalRating'].value_counts()
novice = agreed[agreed['Score'].str.startswith("N") == True]  # ** why some nan in 'score' column
data = novice[['Username', 'Testid', 'Version', 'Response', 'Score']]
data2 = data.dropna(
    subset=['Response'])  # ***** why some missing response prompts still gets assigned a prompt-level score??

# data['Score'].value_counts().plot(kind='bar')
# data['Score'].value_counts()
data = data[data.Score != 'N-UR']
data['Score'].value_counts()  # shortage of 'I-NE' cases
data.to_csv('novice_2018.csv')

frames = [pw19]
result = pd.concat(frames)
agreed = result[result['R1'] == result['R2']]
# agreed['Score'].value_counts().plot(kind='bar')
# agreed['Score'].value_counts()
# agreed['FinalRating'].value_counts().plot(kind='bar')
# agreed['FinalRating'].value_counts()
novice = agreed[agreed['Score'].str.startswith("N") == True]  # ** why some nan in 'score' column
data = novice[['Username', 'Testid', 'Version', 'Response', 'Score']]
data2 = data.dropna(
    subset=['Response'])  # ***** why some missing response prompts still gets assigned a prompt-level score??

# data['Score'].value_counts().plot(kind='bar')
# data['Score'].value_counts()
data = data[data.Score != 'N-UR']
data['Score'].value_counts()  # shortage of 'I-NE' cases
data.to_csv('novice_2019.csv')