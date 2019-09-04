import pickle
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

def formatHeadline(hl):
    hl=re.sub('[^a-zA-Z]',' ',hl)
    hl=hl.lower()
    hl=hl.split()
    ps=PorterStemmer()
    hl=[ps.stem(word) for word in hl if word not in set(stopwords.words('english'))]
    hl=' '.join(hl)
    return hl

pickle_in=open("HeadlinesClassifier.pickle",'rb')
classifier=pickle.load(pickle_in)
cv=pickle.load(pickle_in)

new_headline = 'Doctors find a cure for Cancer'
new_headline= formatHeadline(new_headline)
 
test_corpus = []
test_corpus.append(new_headline)
X_new_test = cv.transform(test_corpus).toarray()
prediction = classifier.predict(X_new_test)
