#Importing Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

train = pd.read_csv('titanic_train.csv')

#print(train.shape)
#print(train.size)
#train.describe()

#Data visulaisation
'''plt.figure(figsize = (10,8))
train['Age'].hist(bins =70)
plt.show()

train['Survived'].hist(bins=2)
plt.show()

val= True
'''

#changing the string variables to binary format and treating missing values

train['Sex'] = train['Sex'].apply(lambda sex:1 if sex =='male' else 0)
train['Age'] = train['Age'].fillna(train['Age'].mean())
train['Fare'] = train['Fare'].fillna(train['Fare'].mean())


test = pd.read_csv('test.csv')
test['Sex'] = test['Sex'].apply(lambda sex:1 if sex =='male' else 0)
test['Age'] = test['Age'].fillna(test['Age'].mean())
test['Fare'] = test['Fare'].fillna(test['Fare'].mean())


#importing the necessary module
from sklearn.linear_model import LogisticRegression

survived = train['Survived'].values
train_data = train.drop(labels = 'Survived', axis =1)
print (train.shape)
print(train_data.shape)
print(test.shape)

model = LogisticRegression()
model.fit(train_data, survived)

LogisticRegression()

predict = model.predict(test)


