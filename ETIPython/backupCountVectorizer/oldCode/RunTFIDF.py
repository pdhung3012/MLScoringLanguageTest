from sklearn.feature_extraction.text import TfidfVectorizer
corpus = [
     'This is the first document.',
     'This document is the second document.',
     'And this is the third one.',
     'Is this the first document?',
]
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(corpus)
arrFeatureNames=vectorizer.get_feature_names()
print('names: '+str(arrFeatureNames))
for i in range(len(arrFeatureNames)):
     print(str(i)+' '+arrFeatureNames[i])
print(X[0][0])
