# _______ is a program that uses a neural network to predict the precipitable stable water isotopes
# Developed by: Jaxton Gray
# Written in Python 3.11.6

# Currently I am only going to be trying to predict the d18O values

# Note for Jax, currently working in a virtual environment called ML_Thesis


# Importing libraries
import numpy as np
import pandas as pd


# Tensorflow libraries
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import RootMeanSquaredError
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split # For splitting the data into training and testing sets, temporarily using this until I can figure out how to use the tensorflow version
from sklearn.preprocessing import StandardScaler

# Importing data
data = pd.read_csv("Isoscape_Data.csv")
# Remove station name and output features that I am not looking at
data = data.drop(['Station', 'dex', 'H2avg'], axis=1)
#Convert the Date column to a datetime object
data['Date'] = pd.to_datetime(data['Date'])

# Separate the data into input and output features
features = data.drop(['O18Avg'], axis=1)

# Extract year, month, day, and day of week
features['year'] = features['Date'].dt.year
features['month'] = features['Date'].dt.month
features['day'] = features['Date'].dt.day

# Now drop the original 'date' column
features.drop('Date', axis=1, inplace=True)

num_features = features.shape[1]
y = data['O18Avg']

# Normalize the data
scaler = StandardScaler()
features = scaler.fit_transform(features)
features = pd.DataFrame(features)


# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, y, test_size=0.2, random_state=42) # I am not sure if I am going to need to use the random_state parameter

# Creating the model
model = Sequential()
model.add(layers.InputLayer(input_shape=(num_features,1)))
model.add(layers.LSTM(64))
model.add(layers.Dense(8, activation='relu'))
model.add(layers.Dense(1, activation='linear'))

# Compiling the model
model.compile(optimizer=Adam(learning_rate=0.001), loss=MeanSquaredError(), metrics=[RootMeanSquaredError()])
# Note: I am not sure if I am going to need to change the learning rate or not

# Creating a checkpoint to save the best model
checkpoint = ModelCheckpoint("model/", save_best_only=True)

# Training the model
model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.2, callbacks=[checkpoint])

# Callback to load the best model
#model.load_weights("best_model.hdf5")

# Displaying the accuracy of the model
print("The accuracy of the model is: ", model.evaluate(X_test, y_test))

# Save the results of the X_test and y_test and y_pred to a csv file
y_pred = model.predict(X_test)
# Convert y_pred to a DataFrame
y_pred_df = pd.DataFrame(y_pred, columns=['y_pred'])

# Select columns of Alt, Lat, Lon, Date, Pre
columns = data[['Alt', 'Lat', 'Long', 'Date', 'Precipitation (kg/m^2/s)', 'Temperature (K)']]

# Concatenate all DataFrames
results = pd.concat([columns, y_test, y_pred_df], axis=1)

# Save the DataFrame to a CSV file
results.to_csv('results.csv', index=False)
