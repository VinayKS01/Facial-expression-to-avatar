import os  
import numpy as np 
import cv2 
from tensorflow.keras.utils import to_categorical
from keras.layers import Input, Dense 
from keras.models import Model

# Initialize variables
is_init = False
label = []
dictionary = {}
c = 0

# Load and concatenate data
for i in os.listdir():
    if i.endswith(".npy") and not i.startswith("labels"):
        if not is_init:
            X = np.load(i)
            size = X.shape[0]
            y = np.array([i.split('.')[0]] * size).reshape(-1, 1)
            is_init = True
        else:
            X = np.concatenate((X, np.load(i)))
            y = np.concatenate((y, np.array([i.split('.')[0]] * size).reshape(-1, 1)))

        label.append(i.split('.')[0])
        dictionary[i.split('.')[0]] = c  
        c += 1

# Convert labels to numerical and then to categorical
y = np.vectorize(dictionary.get)(y.flatten())
y = to_categorical(y)

# Shuffle the data
indices = np.arange(X.shape[0])
np.random.shuffle(indices)
X = X[indices]
y = y[indices]

# Define the model
input_shape = (X.shape[1],)
ip = Input(shape=input_shape)

m = Dense(512, activation="relu")(ip)
m = Dense(256, activation="relu")(m)
op = Dense(y.shape[1], activation="softmax")(m)

model = Model(inputs=ip, outputs=op)

# Compile the model
model.compile(optimizer='rmsprop', loss="categorical_crossentropy", metrics=['acc'])

# Train the model with a validation split
model.fit(X, y, epochs=50, validation_split=0.2)

# Save the model and labels
model.save("model.h5")
np.save("labels.npy", np.array(label))
