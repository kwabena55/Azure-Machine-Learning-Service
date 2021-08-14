import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
dataframe=pd.read_csv('C:/Users/User/Desktop/Data/diabetes.csv')

cols=['Pregnancies', 'PlasmaGlucose', 'DiastolicBloodPressure',
       'TricepsThickness', 'SerumInsulin', 'BMI', 'DiabetesPedigree', 'Age',
       'Diabetic']

dataframe=dataframe[cols]


X_df=dataframe[['Pregnancies', 'PlasmaGlucose', 'DiastolicBloodPressure','TricepsThickness', 'SerumInsulin', 'BMI', 'DiabetesPedigree', 'Age']]
Y_df=dataframe['Diabetic']

X=X_df.values
Y=Y_df.values

scaler=MinMaxScaler(feature_range=(0,1))
rescaledX=scaler.fit_transform(X)

test_size=0.33
seed=7
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=test_size,random_state=seed)

model=LogisticRegression(max_iter=100000)
model.fit(X_train,Y_train)
# result=model.score(X_test,Y_test)
# result
