# -*- coding: utf-8 -*-
"""
Attention model code is based from https://www.kaggle.com/sermakarevich/hierarchical-attention-network
and is based on Yang et al. 2016 paper titled "Hierarchical Attention Networks for Document Classification"
@author: Hung
"""
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import Input
from tensorflow.keras.layers import MaxPool2D, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Layer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, LSTM, Conv1D, MaxPooling1D, Dropout, Activation, Conv2D, Reshape, \
    MaxPooling2D
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import Bidirectional
from tensorflow.keras import initializers, regularizers, constraints
from tqdm.notebook import tqdm
import tensorflow.keras.backend as K
import tensorflow.compat.v1 as tf
# from textblob import TextBlob
import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import LabelBinarizer
import math
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from multiVectors.utils import createDir
from sklearn.model_selection import KFold
from tokenize import tokenize, untokenize, NUMBER, STRING, NAME, OP
from nltk import word_tokenize
from nltk.stem.snowball import SnowballStemmer
# import google-core-.es as lemEsp
from es_lemmatizer import lemmatize
from cucco import Cucco

import spacy
import ftfy
# from spacy.es import Spanish
# nlp = Spanish()

nlp = spacy.load("es_core_news_sm")
nlp.add_pipe(lemmatize, after="tagger")
# nlp.

sbEsp = SnowballStemmer('spanish')

normEsp = Cucco(language='es')
norms = ['remove_stop_words', 'replace_punctuation', 'remove_extra_whitespaces']


cwd = os.getcwd()
os.chdir(cwd)
# base_directory = cwd
base_directory = '/home/hung/git/resultETI/SpanishRater/'
print(base_directory)

# directory = cwd + '/Data'
# clean_directory = cwd + '/Clean_Data'
# directory = base_directory + '/Data'
inputCsvs_directory = base_directory + '/inputCsvs/'
normalize_directory = base_directory + '/inputTextNormalize/'

createDir(inputCsvs_directory)
createDir(normalize_directory)


for filename in os.listdir(inputCsvs_directory):
    print(filename)
    typeName = filename.replace('_text.csv', '')

    df = pd.read_csv(inputCsvs_directory + "/" + filename)
    listNormalize=['no,testId,score,response']
    for i in range(len(df['response'])):
        strItem=str(df['response'][i])
        try:
            strItem=' '.join([sbEsp.stem(item) for item in (strItem).split(' ')])
            doc = nlp(strItem)
            lstItem=[]
            for token in doc:
                lstItem.append(str(token.lemma_))
            strItem=' '.join(lstItem)
            strItem=ftfy.fix_encoding(strItem)
        except:
            print('Error: ',strItem)

        strLine='{},{},{},{}'.format(str(df['no'][i]),str(df['testId'][i]),str(df['score'][i]),strItem)
        listNormalize.append(strLine)

    fpLogTotal = open(normalize_directory + filename, 'w')
    fpLogTotal.write('\n'.join(listNormalize))
    fpLogTotal.close()



















