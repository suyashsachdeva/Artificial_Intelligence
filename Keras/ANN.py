# First ANN that I made

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Activation, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import categorical_crossentropy

model = Sequential([
    Dense(units=16, input_shape=(1,), activation='relu'),
    Dense(units=10000, activation='relu'),
    Dense(units=10000, activation='relu'),
    Dense(units=10000, activation='relu'),
    Dense(units=1000, activation='softmax')
])

model.summary()