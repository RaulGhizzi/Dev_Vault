import os
os.system('clear')

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.formula.api as sm
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import train_test_split
from sklearn import tree
from sklearn.metrics import accuracy_score







#Import Train Dataset
dataset_train = pd.read_csv("train.csv")
x_train = dataset_train.iloc[:,2:12].values
y_train = dataset_train.iloc[:,1].values                            
#Encoding Categorical data and cleaning missing data
labelencoder_x = LabelEncoder() 
x_train[:, 1] = labelencoder_x.fit_transform(x_train[:, 1].astype("str")) 
x_train[:, 2] = labelencoder_x.fit_transform(x_train[:, 2].astype("str")) 
x_train[:, 6] = labelencoder_x.fit_transform(x_train[:, 6].astype("str")) 
imputer = Imputer(missing_values = 574, strategy = 'mean', axis = 1)
imputer.fit(x_train[:,6])
x_train[:, 6] = imputer.transform(x_train[:, 6])
x_train[:, 8] = labelencoder_x.fit_transform(x_train[:, 8].astype("str"))
imputer = Imputer(missing_values = 147, strategy = 'mean', axis = 1)
imputer.fit(x_train[:,8])
x_train[:, 8] = imputer.transform(x_train[:, 8]) 
x_train[:, 9] = labelencoder_x.fit_transform(x_train[:, 9].astype("str")) 
imputer = Imputer(missing_values = 3, strategy = 'mean', axis = 1)
imputer.fit(x_train[:,9])
x_train[:, 9] = imputer.transform(x_train[:, 9])
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 1)
imputer.fit(x_train[:,:])
x_train[:, :] = imputer.transform(x_train[:, :])
onehotencoder = OneHotEncoder(categorical_features = [0])
x_train = onehotencoder.fit_transform(x_train).toarray()

#Feature Scaling
sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train)


#Import Test Dataset
dataset_test = pd.read_csv("test.csv")
x_test = dataset_test.iloc[:,1:11].values                          
#Encoding Categorical data and cleaning missing data
labelencoder_x = LabelEncoder() 
x_test[:, 1] = labelencoder_x.fit_transform(x_test[:, 1].astype("str")) 
x_test[:, 2] = labelencoder_x.fit_transform(x_test[:, 2].astype("str")) 

x_test[:, 6] = labelencoder_x.fit_transform(x_test[:, 6].astype("str")) 
imputer = Imputer(missing_values = 574, strategy = 'mean', axis = 1)
imputer.fit(x_test[:,6])
x_test[:, 6] = imputer.transform(x_test[:, 6])

x_test[:, 8] = labelencoder_x.fit_transform(x_test[:, 8].astype("str"))
imputer = Imputer(missing_values = 147, strategy = 'mean', axis = 1)
imputer.fit(x_test[:,8])
x_test[:, 8] = imputer.transform(x_test[:, 8]) 

x_test[:, 9] = labelencoder_x.fit_transform(x_test[:, 9].astype("str")) 
imputer = Imputer(missing_values = 3, strategy = 'mean', axis = 1)
imputer.fit(x_test[:,9])
x_test[:, 9] = imputer.transform(x_test[:, 9])

imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 1)
imputer.fit(x_test[:,:])
x_test[:, :] = imputer.transform(x_test[:, :])

onehotencoder = OneHotEncoder(categorical_features = [0])
x_test = onehotencoder.fit_transform(x_test).toarray()          



#Feature Scaling
sc_x = StandardScaler()
x_test = sc_x.fit_transform(x_test)





#==============================================================================
#Start Elimination (if max(P-value) in x > sl: remove columns else model is ready)
##Significance_level
#sl = 0.05
# 
#==============================================================================

#Select Columns to keep
x_train_opt = x_train[:,[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]]
regressor_ols = sm.OLS(endog = y_train, exog = x_train_opt).fit()
regressor_ols.summary()

x_train_opt = x_train[:,[0, 1, 2, 4, 5, 6, 7, 8, 9, 10, 11]]
regressor_ols = sm.OLS(endog = y_train, exog = x_train_opt).fit()
regressor_ols.summary()

x_train_opt = x_train[:,[0, 1, 2, 4, 5, 6, 8, 9, 10, 11]]
regressor_ols = sm.OLS(endog = y_train, exog = x_train_opt).fit()
regressor_ols.summary()

x_train_opt = x_train[:,[0, 2, 4, 5, 6, 8, 9, 10, 11]]
regressor_ols = sm.OLS(endog = y_train, exog = x_train_opt).fit()
regressor_ols.summary()

x_train_opt = x_train[:,[0, 2, 4, 5, 6, 8, 10, 11]]
regressor_ols = sm.OLS(endog = y_train, exog = x_train_opt).fit()
regressor_ols.summary()

x_train_opt = x_train[:,[0, 2, 4, 5, 6, 8, 11]]
regressor_ols = sm.OLS(endog = y_train, exog = x_train_opt).fit()
regressor_ols.summary()

x_train_opt = x_train[:,[0, 2, 4, 5, 6, 11]]
regressor_ols = sm.OLS(endog = y_train, exog = x_train_opt).fit()
regressor_ols.summary()

x_train_opt = x_train[:,[0, 2, 4, 5, 11]]
regressor_ols = sm.OLS(endog = y_train, exog = x_train_opt).fit()
regressor_ols.summary()

x_test_opt = x_test[:,[0, 2, 4, 5, 6, 11]]



#Spliting Dataset
x_train_2, x_test_2, y_train_2, y_test_2 = train_test_split(x_train_opt, y_train, test_size = 0.2, random_state = 0)

predictor = tree.DecisionTreeClassifier()
predictor = predictor.fit(x_train_2, y_train_2)
prediction = predictor.predict(x_test_2)
accuracy = accuracy_score(y_test_2, prediction)


