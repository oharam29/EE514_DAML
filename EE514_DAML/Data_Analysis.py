import numpy as np
import pandas as pd
import nltk as nk
import seaborn as sns 
import csv

from collections import Counter
from matplotlib import pyplot as plt

print("Running Data_Analysis:")
print("//----------------------------------------------")


train = pd.read_csv('C:/Users/Nova6/Desktop/College/MCTY/EE514_DAML/csv/train.csv')
nk.download("stopwords")
train_reader = csv.reader(open('C:/Users/Nova6/Desktop/College/MCTY/EE514_DAML/csv/train.csv', encoding="utf8"), delimiter=',')


top20stopwords = Counter(" ".join(train['headline']).split()).most_common(20)
print(top20stopwords)
'''
L = [top20stopwords]
x=()
y=()
for t in L:
	x + ([s[0] for s in t])
	y + (s[1] for s in t)



plt.barh(y, x, align='center', alpha=0.5)
plt.xlabel('Count of word')
plt.title('Most common words - stopwords included')
plt.savefig('C:/Users/Nova6/Desktop/College/MCTY/EE514_DAML/figs/BarChartStopWOrd.png')
plt.show()
'''
list_of_stopwords = nk.corpus.stopwords.words('english')

remove_stopwords = [word for word in " ".join(train['headline']).split() if word not in list_of_stopwords]

top20words = Counter(" ".join(remove_stopwords).split()).most_common(20)
print(top20words)

#comparing the headline length
fakeLength = np.array([])
realLength = np.array([])

for line in train_reader:
	if(line[1] == '1'):
		fakeLength = np.append(fakeLength, len(line[0]))
	else:
		realLength = np.append(realLength,  len(line[0]))

#show stats of headline length
print("//----------------------------------------------")
print("Details of fake headline:")
print("Mean fake length:")  
print(np.mean(fakeLength)) 
print("Max fake length:") 
print(np.max(fakeLength))
print("Min fake length:") 
print(np.min(fakeLength))

print("//----------------------------------------------")
print("Details of real headline:")
print("Mean real length:")
print(np.mean(realLength))
print("Max real length:")
print(np.max(realLength))
print("Min real length:")
print(np.min(realLength))

#plot to boxplot
sns.set(style='whitegrid')
fakePlot = sns.boxplot(data=fakeLength, orient='h')
fakePlot.set(xlabel='Fake Headline Length', title='Fake Headlines')
plt.savefig('C:/Users/Nova6/Desktop/College/MCTY/EE514_DAML/figs/LengthOfFakeHeadlinePlot.png')
plt.show()

realPlot = sns.boxplot(data=realLength, orient='h')
realPlot.set(xlabel= 'Real Headline Length', title='Real Headline')
plt.savefig('C:/Users/Nova6/Desktop/College/MCTY/EE514_DAML/figs/LengthOfRealHeadlinePlot.png')
plt.show()

print("Finshed Data_Analysis:")
print("//----------------------------------------------")
