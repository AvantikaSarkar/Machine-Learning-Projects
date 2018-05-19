''' 
	Linear Regression without and with a few new varibles.
	Performs poorly as it is a linear model and the data inter-dependencies are non-linear
	Best possible RMSE score = ~4650
''' 



import pandas as pd

df = pd.read_csv('train.csv', sep =',', header =0)
df = df.fillna(0)
df = df.iloc[0:200000]

#new variable : user purchase count for training data
user_purchase_count = {}

for user_id in df['User_ID']:
    if user_id in user_purchase_count:
        user_purchase_count[user_id] +=1
    else:
        user_purchase_count[user_id] =1

mylist =[]
for uid in df['User_ID']:
    mylist.append(user_purchase_count[uid])

se = pd.Series(mylist)
df['user_pur_count'] = se.values

#new variable : total purchase amount of a user
tot_amt_list = []
for user_id in df['User_ID']:
   tot_amt_list.append(df.loc[df['User_ID'] == user_id, 'Purchase'].sum())
    
tot_amt_se = pd.Series(tot_amt_list)
df['tot_amt'] = tot_amt_se.values

#new variable : avg expenditure of user

df['avg_expd'] = df['tot_amt']/df['user_pur_count']

#changing categorical values to numeric values for linear reg

#using Label Encoder

from sklearn.preprocessing import LabelEncoder
 
le = LabelEncoder()
df['User_ID'] = le.fit_transform(df['User_ID'])

le = LabelEncoder()
df['Product_ID'] = le.fit_transform(df['Product_ID'])

le = LabelEncoder()
df['Gender'] = le.fit_transform(df['Gender'])

le = LabelEncoder()
df['Age'] = le.fit_transform(df['Age'])

le = LabelEncoder()
df['City_Category'] = le.fit_transform(df['City_Category'])

le = LabelEncoder()
df['Stay_In_Current_City_Years'] = le.fit_transform(df['Stay_In_Current_City_Years'])


'''splitting dataset into test and train data'''
from sklearn.model_selection import train_test_split

train, test = train_test_split(df,test_size = 0.3, shuffle = True)

# Building the Linear_Regression model
from sklearn.linear_model import LinearRegression
import numpy as np

x = np.array([train['User_ID'], train['Product_ID'], train['Gender'],  train['Age'], train['Occupation'], train['City_Category'],train['user_pur_count'], train['tot_amt'], train['avg_expd']])
x = x.transpose()
y = np.array(train['Purchase'])

x_test = np.array([test['User_ID'], test['Product_ID'], test['Gender'],  test['Age'], test['Occupation'], test['City_Category'],test['user_pur_count'], test['tot_amt'], test['avg_expd']])
x_test = x_test.transpose()
y_test = np.array(test['Purchase'])


reg = LinearRegression()
reg = reg.fit(x,y)

y_pred = reg.predict(x_test)

print("Sample predicted output :")
print(y_pred.head(10))

from sklearn.metrics import mean_squared_error

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 =reg.score(x_test,y_test)
print("Root-mean-squared-error :")
print('rmse =', rmse)
print('Accuracy of the Linear Regression Model :', round(r2*100, 2),"%")






     