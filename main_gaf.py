import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pyts.image import GramianAngularField
import os

filename = 'Train_file/mitbih_train.csv'

# Create the folders
os.makedirs('dataset/normal', exist_ok=True)
os.makedirs('dataset/anomaly', exist_ok=True)

gaf = GramianAngularField(method='summation')

print("Searching for both Normal and Anomaly heartbeats...")

# Load a larger chunk to ensure we hit anomalies (usually found after row 70,000)
# But we'll use a trick: load 500 rows from the start and 500 from the end
df_start = pd.read_csv(filename, header=None, nrows=500)
df_end = pd.read_csv(filename, header=None, skiprows=75000, nrows=500)
df = pd.concat([df_start, df_end])

X = df.iloc[:, :-1].values  
y = df.iloc[:, -1].values

for i in range(len(X)):
    signal = X[i].reshape(1, -1)
    label = y[i]
    
    image_matrix = gaf.fit_transform(signal)
    
    # 0 is normal, anything else is anomaly
    folder = 'normal' if label == 0 else 'anomaly'
    
    plt.imsave(f'dataset/{folder}/heartbeat_{i}.png', image_matrix[0], cmap='rainbow')

print("--- DONE! ---")
print("Check your 'anomaly' folder now. It should have images!")