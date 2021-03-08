import pandas as pd
import pickle

from sklearn.feature_extraction import text
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics

print("Running Supervised_Class:")
print("//----------------------------------------------")


#point at files
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

#TFIDF on files
tfidf = TfidfVectorizer()
tfidf_Train = tfidf.fit_transform(X_train)
tfidf_Val = tfidf.transform(X_val)


#Naive Bayes on training data
supervised = MultinomialNB()
supervised.fit(tfidf_Train, Y_train)
predictions = supervised.predict(tfidf_Val)

print("Accuracy of MultinomialNB: ")
print(metrics.accuracy_score(Y_val, predictions))

correct = 0
incorrect = 0
#check the predictions
for i, pre, lab in zip(X_val, predictions, Y_val):
	if pre != lab:
		incorrect+=1
	else:
		correct+=1

print("Number of correct predictions: " + str(correct))
print("Number of incorrect predications: " + str(incorrect))

with open('C:/Users/Nova6/Desktop/College/MCTY/EE514_DAML/pickle/save_model.pickle', 'wb') as file:
	pickle.dump(supervised, file)

print("Finished Supervised_Class:")
print("//----------------------------------------------")

