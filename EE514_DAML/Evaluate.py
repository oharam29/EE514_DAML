import pandas as pd 
import pickle
import matplotlib.pyplot as plt

from sklearn.feature_extraction import text
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve, roc_auc_score

print("Running Evaluate:")
print("//----------------------------------------------")


train = 'C:/Users/Nova6/Desktop/College/MCTY/EE514_DAML/csv/train.csv'
test = 'C:/Users/Nova6/Desktop/College/MCTY/EE514_DAML/csv/testing.csv'

X_train = pd.read_csv(train)['headline']
Y_train = pd.read_csv(train)['is_sarcastic']

X_test = pd.read_csv(test)['headline']
Y_test = pd.read_csv(test)['is_sarcastic']

#tfidf on test
tfidf = TfidfVectorizer()
tfidf_Train = tfidf.fit_transform(X_train)
tfidf_Test = tfidf.transform(X_test)

#open with pickle
with open('C:/Users/Nova6/Desktop/College/MCTY/EE514_DAML/pickle/save_model_LogisticRegression.pickle', 'rb') as file:
	best_model = pickle.load(file)

prediction = best_model.predict(tfidf_Test)
print("The accuracy of the model on the test data is:")
print(metrics.accuracy_score(Y_test, prediction)) 

print(classification_report(Y_test, prediction))

print("Confusion Matrix of model:")
print(metrics.confusion_matrix(Y_test, prediction))

#roc curve and area under curve value
fpr, tpr, thresholds = roc_curve(Y_test, prediction)
Area_under_curve = roc_auc_score(Y_test, prediction)

plt.plot(fpr, tpr, label='ROC CURVE (Area Under the Curve = %0.3f' % Area_under_curve)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.title('ROC')
plt.legend(loc='lower right')
plt.savefig('C:/Users/Nova6/Desktop/College/MCTY/EE514_DAML/figs/ROC.png')
plt.show()

print("Finished Evaluate:")
print("//----------------------------------------------")
