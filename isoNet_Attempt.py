# # # # #
#
#   .----------------.  .----------------.  .----------------.  .-----------------. .----------------.  .----------------. 
#   | .--------------. || .--------------. || .--------------. || .--------------. || .--------------. || .--------------. |
#   | |     _____    | || |    _______   | || |     ____     | || | ____  _____  | || |  _________   | || |  _________   | |
#   | |    |_   _|   | || |   /  ___  |  | || |   .'    `.   | || ||_   \|_   _| | || | |_   ___  |  | || | |  _   _  |  | |
#   | |      | |     | || |  |  (__ \_|  | || |  /  .--.  \  | || |  |   \ | |   | || |   | |_  \_|  | || | |_/ | | \_|  | |
#   | |      | |     | || |   '.___`-.   | || |  | |    | |  | || |  | |\ \| |   | || |   |  _|  _   | || |     | |      | |
#   | |     _| |_    | || |  |`\____) |  | || |  \  `--'  /  | || | _| |_\   |_  | || |  _| |___/ |  | || |    _| |_     | |
#   | |    |_____|   | || |  |_______.'  | || |   `.____.'   | || ||_____|\____| | || | |_________|  | || |   |_____|    | |
#   | |              | || |              | || |              | || |              | || |              | || |              | |
#   | '--------------' || '--------------' || '--------------' || '--------------' || '--------------' || '--------------' |
#   '----------------'  '----------------'  '----------------'  '----------------'  '----------------'  '----------------' 
#
# # # # #

import numpy as np
import pandas as pd

# Tensorflow libraries
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, InputLayer
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import RootMeanSquaredError, MeanAbsoluteError

### First we will make functions that will build the model, then we will load the data in, and finally we will train the model ###


# Designing the model
# Will require the following parameters:
#   - window_size: number of time steps to look back on
#   - numFeatures: number of features in the dataset 
#   - numNeurons: number of neurons in the hidden layer HYPERPARAMETER
#   - lrate: learning rate of the Adam optimizer HYPERPARAMETER
def build_model(numFeatures, windowSize, numNeurons, lrate):
    model = Sequential()
    model.add(InputLayer(input_shape=(windowSize, numFeatures,)))
    model.add(LSTM(numNeurons))
    model.add(Dense(numNeurons))
    model.add(Dense(1))

    model.compile(
        loss=MeanAbsoluteError(),
        optimizer=Adam(learning_rate=lrate),
        metrics=[RootMeanSquaredError()]
        )

    return model
    

# Load in the data, separate the features and labels
data = pd.read_csv(r'Isoscape_Data.csv')
features = data.drop(['H2avg', 'dex', 'O18Avg', 'Station'], axis=1)
labels = data['O18Avg']

# Adjusting the date time to be int values and separating the year, month and day
features['Date'] = pd.to_datetime(features['Date'])
features['Year'] = features['Date'].dt.year
features['Month'] = features['Date'].dt.month
features['Day'] = features['Date'].dt.day
features = features.drop(['Date'], axis=1)
numFeatures = len(features.columns)

# Convert into numpy arrays
features = features.to_numpy()
labels = labels.to_numpy()

# Split the data into training and testing sets
X_train = features[:int(len(features)*0.8)]
y_train = labels[:int(len(labels)*0.8)]
X_test = features[int(len(features)*0.8):]
y_test = labels[int(len(labels)*0.8):]

# Combine into tensorflow datasets
trainData = tf.data.Dataset.from_tensor_slices((X_train, y_train))
testData = tf.data.Dataset.from_tensor_slices((X_test, y_test))

# Window the data
window_size = 5
trainData = trainData.window(window_size, shift=1, drop_remainder=True)
testData = testData.window(window_size, shift=1, drop_remainder=True)

# Flatten the datasets
trainData = trainData.flat_map(lambda window: window.batch(window_size))
testData = testData.flat_map(lambda window: window.batch(window_size))

# Build the model using the function
model = build_model(numFeatures, window_size, 64, 0.001)

# Train the model
early_stopping = EarlyStopping(patience=5)
model.fit(trainData, epochs=50, validation_data=testData, callbacks=[early_stopping])