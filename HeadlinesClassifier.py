import pickle
import pandas as pd
dataset = pd.read_csv("uci-news-aggregator.csv") 

import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
corpus=[]

def formatHeadline(hl):
    hl=re.sub('[^a-zA-Z]',' ',hl)
    hl=hl.lower()
    hl=hl.split()
    ps=PorterStemmer()
    hl=[ps.stem(word) for word in hl if word not in set(stopwords.words('english'))]
    hl=' '.join(hl)
    return hl

for i in range(0,70000):           #Change here
    corpus.append(formatHeadline(dataset['TITLE'][i]))
    print(i)
    

from sklearn.feature_extraction.text import CountVectorizer
cv= CountVectorizer(max_features=1500)
X=cv.fit_transform(corpus).toarray()
y=dataset.iloc[:,4].values

from sklearn.preprocessing import LabelEncoder
labelencoder=LabelEncoder()
y=labelencoder.fit_transform(y)
y=y[0:70000]       #Change here

from sklearn.cross_validation import train_test_split
X_train,X_test,y_train,y_test= train_test_split(X,y,test_size=0.2)

from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)

from sklearn.naive_bayes import GaussianNB
classifier=GaussianNB()
classifier.fit(X_train,y_train)

with open("HeadlinesClassifier.pickle",'wb') as f:
    pickle.dump(classifier,f)
    pickle.dump(cv,f)

pickle_in=open("HeadlinesClassifier.pickle",'rb')
classifier=pickle.load(pickle_in)
 

y_pred=classifier.predict(X_test)


from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)


new_headline = 'General Motors expands ignition switch recall to 2.6 million cars'
new_headline= formatHeadline(new_headline)
 
test_corpus = []
test_corpus.append(new_headline)
X_new_test = cv.transform(test_corpus).toarray()
prediction = classifier.predict(X_new_test)
