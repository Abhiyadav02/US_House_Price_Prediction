# -*- coding: utf-8 -*-
"""
Created on Wed Sep 13 21:37:32 2023

@author: abhis
"""


'''
1. Importing required libraries
'''
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
#%matplotlib inline
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score, train_test_split, GridSearchCV


'''
2. Importing Dataset
'''
df = pd.read_csv("F:/ML-DS-PYTHON/My_projects/US_House_Price_Prediction/houseprice.csv")
df.head()
df.shape
df.describe
df.info


'''
3.1 Finding columns with missing values and their percent missing
'''

df.isnull().sum()
missing_val = df.isnull().sum().sort_values(ascending = False)
missing_val = pd.DataFrame(data = df.isnull().sum().sort_values(ascending = False), columns = ['MissingVal_Count'])


'''
3.2 Add a new column to the dataframe and fill it with the percentage of missing values
'''
missing_val['Percent'] = missing_val.MissingVal_Count.apply(lambda x: '{:.2f}'.format(float(x)/df.shape[0] * 100))
missing_val = missing_val[missing_val.MissingVal_Count > 0]
missing_val


'''
3.3 Drop columns with high missing values
'''
df = df.drop(['Fence', 'MiscFeature', 'PoolQC', 'FireplaceQu', 'Alley'], axis = 1)

'''
3.4 Drop rows with any missing values
'''
df.dropna(inplace = True)
df.shape
sns.distplot(df.SalePrice)
sns.distplot(np.log(df.SalePrice))
sns.displot(df.SalePrice)
sns.displot(np.log(df.SalePrice))

df['LogOfPrice'] = np.log(df.SalePrice)
df.drop(['SalePrice'], axis = 1, inplace = True)

df.skew().sort_values(ascending = False)


'''
4.1 Set the target and predictors
'''
y = df.LogOfPrice  # Target


'''
4.2 Use only those input features with numeric data type
'''
df_temp = df.select_dtypes(include = ["int64", "float64"])
X = df_temp.drop(['LogOfPrice'], axis = 1)  # Predictor

'''
4.3 Split the dataset into train and test sets
'''
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .25, random_state = 3)

'''
5.1 Fit optimal linear regression line on training data, this performs gradient descent under the hood
'''
lr = LinearRegression()
lr.fit(X_train, y_train)

yr_hat = lr.predict(X_test)

'''
5.2 Evaluate the algorithm with a test set
'''
lr_score = lr.score(X_test, y_test)  # Train Test
print("Accuracy: ", lr_score)


'''
5.3 Cross validation to find 'validate' score across multiple samples, automatically does Kfold stratifying
'''
lr_cv = cross_val_score(lr, X, y, cv = 5, scoring = 'r2')
print('Cross-validation Result: ', lr_cv)
print('R2: ', lr_cv.mean())
