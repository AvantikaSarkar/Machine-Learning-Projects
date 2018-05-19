'''
    XGBoost with feature engineering 
    3 new features were added
    Feature Importance feature of xgboost suggested that the new features had significant importance compared to other features
    RMSE score = ~2600 (accuracy = 72%)
'''





import pandas as pd

df = pd.read_csv('train.csv',sep = ',', header =0)
df = df.iloc[0:100000]
df = df.fillna(0)


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


#feature representing the user_purchase_count

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


#feature representing the number of observations of the user

product_count = {}

for prod_id in df['Product_ID']:
    if prod_id in product_count:
        product_count[prod_id] +=1
    else:
        product_count[prod_id] =1

mylist2 =[]
for pid in df['Product_ID']:
    mylist2.append(product_count[pid])
se2 = pd.Series(mylist2)
df['product_count'] = se2.values


#feature representing the avg purchase price of a product

tot_amt_list = []
for prod_id in df['Product_ID']:
   tot_amt_list.append(df.loc[df['Product_ID'] == prod_id, 'Purchase'].sum())
    
tot_amt_se = pd.Series(tot_amt_list)
df['prod_avg_price'] = tot_amt_se.values/df['product_count']

#test train split
from sklearn.model_selection import train_test_split
train, test = train_test_split(df, test_size = 0.2, shuffle = True )

#
import numpy as np
x_train = np.array([train['User_ID'], train['Product_ID'], train['Gender'],  train['Age'], train['Occupation'], train['City_Category'],train['Stay_In_Current_City_Years'], train['Marital_Status'], train['Product_Category_1'], train['Product_Category_2'], train['Product_Category_3'], train['user_pur_count'],train['product_count'], train['prod_avg_price']])
x_train = x_train.transpose()
y_train = np.array(train['Purchase'])

x_test = np.array([test['User_ID'], test['Product_ID'], test['Gender'],  test['Age'], test['Occupation'], test['City_Category'],test['Stay_In_Current_City_Years'], test['Marital_Status'], test['Product_Category_1'], test['Product_Category_2'], test['Product_Category_3'], test['user_pur_count'],test['product_count'], test['prod_avg_price']])
x_test = x_test.transpose()
y_test = np.array(test['Purchase'])

print("3 new features were added. \n")

#XGBoost

from xgboost import XGBRegressor

model  = XGBRegressor()
model.fit(x_train, y_train)
#print(model.feature_importances_)
y_pred = model.predict(x_test)


from sklearn.metrics import mean_squared_error,r2_score

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print('RMSE value for XGBoost model')
print('RMSE :', rmse)
print('Accuracy :', round(r2_score(y_test, y_pred)*100, 2),"%")



