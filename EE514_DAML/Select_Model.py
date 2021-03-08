import pandas as pd
import pickle

from sklearn.feature_extraction import text
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn import metrics
from sklearn.metrics import classification_report

print("Running Select_Model:")
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


#TFIDF on files
tfidf = TfidfVectorizer()
tfidf_Train = tfidf.fit_transform(X_train)
tfidf_Test = tfidf.transform(X_test)
tfidf_Val = tfidf.transform(X_val)

#models
classifier1 = MultinomialNB()
classifier2 = LogisticRegression(solver='newton-cg', multi_class='multinomial')
classifier3 = RandomForestClassifier(n_estimators=25)
classifier4 = svm.SVC(gamma='scale', probability=True)

#store classifiers and names
list_classifiers = [classifier1,classifier2,classifier3,classifier4]
names = ["MultinomialNB", "LogisticRegression", "RandomForestClassifier", "SVC"]


for i in range(4):
	classifier = list_classifiers[i]
	classifier_name = names[i]

	classifier.fit(tfidf_Train, Y_train)
	prediction = classifier.predict(tfidf_Val)
	accuracy = metrics.accuracy_score(Y_val, prediction)
	

	print("//----------------------------------------------")
	print(classifier_name)

	print("The accuracy of " + classifier_name + " is:")
	print(accuracy)

	print(classification_report(Y_val, prediction, target_names=['Real Headline', 'Fake Headline']))

	#save model
	if(classifier_name == "MultinomialNB"):
		with open('C:/Users/Nova6/Desktop/College/MCTY/EE514_DAML/pickle/save_model_MultinomialNB.pickle', 'wb') as file:
			pickle.dump(classifier, file)

	if(classifier_name == "LogisticRegression"):
		with open('C:/Users/Nova6/Desktop/College/MCTY/EE514_DAML/pickle/save_model_LogisticRegression.pickle', 'wb') as file:
			pickle.dump(classifier, file)

	if(classifier_name == "RandomForestClassifier"):
		with open('C:/Users/Nova6/Desktop/College/MCTY/EE514_DAML/pickle/save_model_RandomForestClassifier.pickle', 'wb') as file:
			pickle.dump(classifier, file)

	if(classifier_name == "SVC"):
		with open('C:/Users/Nova6/Desktop/College/MCTY/EE514_DAML/pickle/save_model_SVC.pickle', 'wb') as file:
			pickle.dump(classifier, file)

	print(classifier_name + "finished")


print("Running Select_Model:")
print("//----------------------------------------------")


'''
	f1_score_macro = metrics.f1_score(Y_val, prediction, average='macro')
	f1_score_weighted = metrics.f1_score(Y_val, prediction, average='weighted')

	print("The macro avg of " + classifier_name + " is:")
	print(f1_score_macro)

	print("The weighted avg of " + classifier_name + " is:")
	print(f1_score_weighted)
'''


