import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

print("Running MLSplit:")
print("//----------------------------------------------")

#path for files
train = 'C:/Users/Nova6/Desktop/College/MCTY/EE514_DAML/csv/train.csv'
test = 'C:/Users/Nova6/Desktop/College/MCTY/EE514_DAML/csv/testing.csv'
val = 'C:/Users/Nova6/Desktop/College/MCTY/EE514_DAML/csv/valid.csv'

file = 'C:/Users/Nova6/Desktop/College/MCTY/EE514_DAML/edit_fake_news.json'
df = pd.read_json(file)

print(df.shape)
print(df.head)


#split off training 
X_training_and_valid, X_test, Y_training_and_valid, Y_test = train_test_split(df['headline'], df['is_sarcastic'], test_size = 0.25, random_state = 117)

#split off validation from training
X_train, X_val, Y_train, Y_val = train_test_split(X_training_and_valid, Y_training_and_valid, test_size=0.20, random_state=117)

#Count the labels in the dataset
print( df['is_sarcastic'].sum()/len(df['is_sarcastic']))
print( Y_train.sum(), len(Y_train)-Y_train.sum(), Y_val.sum(), len(Y_val)-Y_val.sum(), Y_test.sum(), len(Y_test)-Y_test.sum())


#Adds training data to csv 
pd.concat([X_train,  Y_train], axis = 1).to_csv(train,index=False)

#Adds test data to csv
pd.concat([X_test, Y_test], axis=1).to_csv(test, index=False)

#Add valid data to csv
pd.concat([X_val, Y_val], axis=1).to_csv(val, index=False)

'''
xl = pd.ExcelFile('C:/Users/Nova6/Desktop/College/MCTY/EE514_DAML/csv/train.csv')
df2 = xl.parse("testing")
answer = df[1].value_counts()
print(answer)'''

print("Finished MLSplit:")
print("//----------------------------------------------")
