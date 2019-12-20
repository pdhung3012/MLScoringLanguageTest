# -*- coding: utf-8 -*-
"""
Created on Mon Aug 19 14:23:42 2019

@author: ziweizh
"""
import pandas as pd
import os

pw18 = pd.read_csv('/Users/hungphan/git/ETI-Project/AAPPLResults_071119_2018Schema.txt', sep=';')
pw19 = pd.read_csv('/Users/hungphan/git/ETI-Project/AAPPLResults_071119_2019Schema.txt', sep=';')
frames = [pw18, pw19]
result = pd.concat(frames)

agreed = result[result['R1'] == result['R2']]
agreed['Score'].value_counts().plot(kind='bar')
agreed['Score'].value_counts()

agreed['FinalRating'].value_counts().plot(kind='bar')
agreed['FinalRating'].value_counts()

intermediate = agreed[agreed['Score'].str.contains("I")==True] # ** why some nan in 'score' column

data = intermediate[['Response','Score']]
