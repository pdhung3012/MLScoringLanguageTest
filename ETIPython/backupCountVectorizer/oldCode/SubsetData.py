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
agreed['Score'].value_counts().plot(kind='bar')
print(agreed)
# agreed['Score'].value_counts()
# agreed['FinalRating'].value_counts().plot(kind='bar')
# agreed['FinalRating'].value_counts()



# intermediate = agreed[agreed['Score'].str.contains("I") == True]  # ** why some nan in 'score' column
# data = intermediate[['Username', 'Testid', 'Version', 'Response', 'Score']]
# data2 = data.dropna(
#     subset=['Response'])  # ***** why some missing response prompts still gets assigned a prompt-level score??
#
# # data['Score'].value_counts().plot(kind='bar')
# # data['Score'].value_counts()
# data = data[data.Score != 'I-UR']
# data['Score'].value_counts()  # shortage of 'I-NE' cases
# data.to_csv('2018.csv')