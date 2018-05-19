''' 
	Random forest on raw data gave the best results, 
	Parameter tuning was done (mostly trial and error)
	RMSE score = ~1930 (accuracy = 84%)
'''



import pandas as pd

df = pd.read_csv('train.csv',sep = ',', header =0)
df = df.iloc[0:100000]
df = df.fillna(0)

#print(df.dtypes)

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

from sklearn.model_selection import train_test_split
train, test = train_test_split(df, test_size = 0.2, shuffle = True )

import numpy as np
x_train = np.array([train['User_ID'], train['Product_ID'], train['Gender'],  train['Age'], train['Occupation'], train['City_Category'],train['Stay_In_Current_City_Years'], train['Marital_Status'], train['Product_Category_1'], train['Product_Category_2'], train['Product_Category_3']])
x_train = x_train.transpose()
y_train = np.array(train['Purchase'])

x_test = np.array([test['User_ID'], test['Product_ID'], test['Gender'],  test['Age'], test['Occupation'], test['City_Category'],test['Stay_In_Current_City_Years'], test['Marital_Status'], test['Product_Category_1'], test['Product_Category_2'], test['Product_Category_3']])
x_test = x_test.transpose()
y_test = np.array(test['Purchase'])

print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)


#Random forst regressor

from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor(n_estimators = 250, max_depth = 15, random_state = 26)
model.fit(x_train, y_train)
y_pred = model.predict(x_test)
y_train_pred = model.predict(x_train)

from sklearn.metrics import mean_squared_error
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 =  model.score(x_test,y_test)
rmse_train = np.sqrt(mean_squared_error(y_train, y_train_pred))
r2_train =  model.score(x_train,y_train)
print(rmse , r2*100)
print(rmse_train, r2_train)


