from tensorflow.keras.models import Model, Sequential
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import layers
import tensorflow.keras as k
import librosa as lr
import pandas as pd
import numpy as np
import joblib
import time
import os

le = LabelEncoder()

data = joblib.load('data.pkl')
data = data.reset_index()
X = np.expand_dims(data.iloc[:, 1:].values, axis=2)
y = le.fit_transform(data.iloc[:, 0].values)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(X_train.shape)

inputs = k.Input(shape=(20, 1))
x = k.layers.Conv1D(filters=128, kernel_size=3)(inputs)
x = k.layers.BatchNormalization()(x)
x = k.layers.LeakyReLU()(x)
x = k.layers.Conv1D(filters=64, kernel_size=3)(x)
x = k.layers.LeakyReLU()(x)
x = k.layers.MaxPool1D(pool_size=2)(x)
x = k.layers.Dropout(rate=0.3)(x)
x = k.layers.BatchNormalization()(x)
x = k.layers.Conv1D(filters=32, kernel_size=3)(x)
x = k.layers.LeakyReLU()(x)
x = k.layers.Conv1D(filters=16, kernel_size=3)(x)
x = k.layers.MaxPool1D(pool_size=2)(x)
x = k.layers.Dropout(rate=0.3)(x)

x = k.layers.Flatten()(x)
x = k.layers.Dense(units=462)(x)
outputs = k.layers.Softmax()(x)


model = Model(inputs, outputs, name='cnn_model')
model.compile(optimizer=Adam(lr=0.005), loss='sparse_categorical_crossentropy', metrics=['accuracy', 'mae'])
model.fit(X_train, y_train, epochs=100, validation_data=(X_test, y_test))


