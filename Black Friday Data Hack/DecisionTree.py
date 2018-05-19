'''
	Decision tree on raw data wihtout feature selection :
	RMSE score = ~2929 (accuracy = 64%)

	After feature selection :
	no of selected features = 5 out of 12
	RMSE score = ~2700 (accuracy = 68%)

	Methods of features selection used :
	1. Mutual Info regression
	2. SelectFromModel with L1 regularisation

	Decision trees perform much better as it captures the non-linearity of the data
'''




import pandas as pd

df = pd.read_csv('train.csv', sep =',', header =0)
df = df.fillna(0)

#print(df.shape)

#Converting categorical to numerical
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

#print(df.dtype)

from sklearn.model_selection import train_test_split

train, test = train_test_split(df,test_size = 0.3, shuffle = True)


import numpy  as np

X = np.array([df['User_ID'], df['Product_ID'],df['Gender'],  df['Age'], df['Occupation'], df['City_Category'], df['Stay_In_Current_City_Years'], df['Marital_Status'], df['Product_Category_1'],  df['Product_Category_2'],  df['Product_Category_3']])
X =X.transpose()
Y = np.array(df['Purchase'])


from sklearn.feature_selection import VarianceThreshold
vt =  VarianceThreshold(threshold = 2.0)
x1 = vt.fit_transform(x1)
print(X.shape)

#using mutual information regression for feature selection 
from sklearn.feature_selection import mutual_info_regression

mir = mutual_info_regression(x,y)
print('The values for mutual information regression is as follows')
print(mir)


# building model
from sklearn.tree import DecisionTreeRegressor

model = DecisionTreeRegressor(max_depth = 10, max_leaf_nodes = 500)
model.fit(x_train,y_train)

y_pred = model.predict(x_test)

from sklearn.metrics import mean_squared_error,r2_score

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print('RMSE value for decision tree before performing feature selection :')
print('RMSE =', rmse)
print('Accuracy :', round(r2_score(y_test, y_pred)*100, 2),"%")
print("\n")


#using SelectFromModel with L1 regularisation(LassoCV) for feature selction
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LassoCV

est = LassoCV()
sfm = SelectFromModel(est, threshold =0.05)

x_new = sfm.fit_transform(x,y)
n  = x_new.shape[1]
print("Number of features selected by the feature selection model : ", n)
print("\n")


    
#print(x_new.shape)
#sfm.transform(x)

x_train = x_new[0:40000, :]
x_test = x_new[40001:50000,:]

y_train = y[0:40000]
y_test = y[40001:50000]

#print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)

#Decision tree

from sklearn.tree import DecisionTreeRegressor

model = DecisionTreeRegressor(max_depth = 10, max_leaf_nodes = 500)
model.fit(x_train,y_train)

y_pred = model.predict(x_test)

print("Sample predicted output :")
print(y_pred.head(10))

from sklearn.metrics import mean_squared_error,r2_score

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print('RMSE value for decision tree after performing feature selection :')
print('RMSE =', rmse)
print('Accuracy :', round(r2_score(y_test, y_pred)*100, 2),"%")

#from sklearn.tree import export_graphviz
#export_graphviz(model2, out_file ="tree.dot")

