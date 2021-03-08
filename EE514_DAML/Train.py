import numpy as np
import pandas as pd 
import matplotlib as plt 
import seaborn as sns

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction import DictVectorizer


from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

print("Running Train:")
print("//----------------------------------------------")


train = 'C:/Users/Nova6/Desktop/College/MCTY/EE514_DAML/csv/train.csv'
test = 'C:/Users/Nova6/Desktop/College/MCTY/EE514_DAML/csv/testing.csv'
val = 'C:/Users/Nova6/Desktop/College/MCTY/EE514_DAML/csv/valid.csv'

#read in files
X_train = pd.read_csv(train)['headline']
Y_train = pd.read_csv(train)['is_sarcastic']

X_test = pd.read_csv(test)['headline']
Y_test = pd.read_csv(test)['is_sarcastic']

X_val = pd.read_csv(val)['headline']
Y_val = pd.read_csv(val)['is_sarcastic']

#remove the stop words
cv_with_stop_words = CountVectorizer()
cv_wsw_res = cv_with_stop_words.fit_transform(X_train)
print(cv_wsw_res.toarray())

#print(cv_with_stop_words.vocabulary_)

cv_no_stop_words = CountVectorizer(stop_words='english')
cv_nsw_res = cv_no_stop_words.fit_transform(X_train)
print(cv_nsw_res.toarray())

#print(cv_no_stop_words.vocabulary_)


print("Finshed Train:")
print("//----------------------------------------------")
