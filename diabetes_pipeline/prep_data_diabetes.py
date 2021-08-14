# Import libraries
import os
import argparse
import pandas as pd
from azureml.core import Run
from sklearn.preprocessing import MinMaxScaler
dataframe=pd.read_csv('C:/Users/User/Desktop/Data/diabetes.csv')
# Normalize the numeric columns
scaler = MinMaxScaler()
num_cols = ['Pregnancies','PlasmaGlucose','DiastolicBloodPressure','TricepsThickness','SerumInsulin','BMI','DiabetesPedigree']
dataframe[num_cols] = scaler.fit_transform(dataframe[num_cols])
# Get the experiment run context
run = Run.get_context()
# Log processed rows
row_count = (len(dataframe))
run.log('processed_rows', row_count)
# remove nulls
dataframe = dataframe.dropna()
# Log processed rows
row_count = (len(dataframe))
run.log('processed_rows', row_count)

# Save the prepped data
print("Saving Data...")
os.makedirs(experiment_folder, exist_ok=True)
save_path = os.path.join(experiment_folder,'data.csv')
dataframe.to_csv(save_path, index=False, header=True)

# End the run
run.complete()
