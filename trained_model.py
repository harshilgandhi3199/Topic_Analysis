# -*- coding: utf-8 -*-
"""
Created on Mon Mar 23 17:33:23 2020

@author: Harshil
"""

#Importing libraries
import logging
import pandas as pd
import numpy as np
from numpy import random
import nltk
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
from mlxtend.plotting import plot_decision_regions
from nltk.corpus import stopwords
import re
from bs4 import BeautifulSoup

#Importing Dataset -- stack_overflow_data
dataset=pd.read_csv('stack_overflow_data.csv')
dataset = dataset[pd.notnull(dataset['tags'])]
X_data=np.ndarray(dataset.iloc[:,0])
y_data=dataset.iloc[:,1]


#Data preprocessing part
to_be_replaced=re.compile('[/(){}\[\]\|@,;]')
bad_words=re.compile('[^0-9a-z #+_]')
STOPWORDS=set(stopwords.words('english'))

def process_text(text):
    text=BeautifulSoup(text, 'lxml').text #HTML decoding
    text=text.lower()
    text=to_be_replaced.sub(" ",text)
    text=bad_words.sub("",text)
    text=' '.join(word for word in text.split() if word not in STOPWORDS)
    return text

dataset['post']=dataset['post'].apply(process_text)
#print_plot(10)

X=dataset.post
y=dataset.tags

"""
texts=["".join(process_text(text)) for text in X]

from sklearn.feature_extraction.text import CountVectorizer
matrix=CountVectorizer(max_features=1000)
vectors=matrix.fit_transform(texts).toarray()
X_train, X_test, y_train, y_test=train_test_split(vectors, y, test_size=0.3, random_state=42)
"""

#Splitting data into training and test set
X_train, X_test, y_train, y_test=train_test_split(X, y, test_size=0.5, random_state=2)
 
#Applying pipeline - CountVectorizer -> TfidfTransformer -> RandomForestClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfTransformer

clf = Pipeline([('vect', CountVectorizer()),
               ('tfidf', TfidfTransformer()),
               ('clf',RandomForestClassifier(n_estimators=100, criterion="entropy", random_state=0)),
              ])
clf.fit(X_train, y_train)

"""#naive bayes classifier
from sklearn.regre import MultinomialNB
classifier=MultinomialNB()
classifier.fit(X_train, y_train)

#logistic regression
from sklearn.linear_model import LogisticRegression
classifier=LogisticRegression(random_state=0)
classifier.fit(X_train, y_train)"""

#Predicting the test set
from sklearn.metrics import classification_report
y_pred = clf.predict(X_test)

#Analyzing the model performance
print('accuracy %s' % accuracy_score(y_pred, y_test))
print(classification_report(y_test, y_pred))

plot_decision_regions(X_data, y_data, clf=clf, legend=2)
plt.xlabel('Post')
plt.ylabel('Tags')
