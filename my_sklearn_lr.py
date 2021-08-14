# Load the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import sklearn
import joblib
import datetime
from azureml.core import Run
run = Run.get_context()
# Load Boston data
from sklearn.datasets import load_boston
boston_dataset = load_boston()
# Train test split data
from sklearn.model_selection import train_test_split
num_Rooms_Train, num_Rooms_Test, med_price_Train, med_Price_Test = train_test_split(boston_dataset.data[:,5].reshape(-1,1), boston_dataset.target.reshape(-1,1))
# Create Linear Regression model
from sklearn.linear_model import LinearRegression
price_room = LinearRegression()
price_room.fit (num_Rooms_Train,med_price_Train)
