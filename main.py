import scipy.io
import glob
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def swish(x):
    return x * tf.nn.sigmoid(x)

data = {
    'a': [],
    'b': [],
    'c': [],
    'd': [],
    'e': [],
    'f': [],
    'g': []
}

for mat in glob.glob("voweldata/*.mat"):
    tmp = scipy.io.loadmat(mat)

    if 'a' in tmp:
        data['a'].append(tmp['a'])
    elif 'b' in tmp:
        data['b'].append(tmp['b'])
    elif 'c' in tmp:
        data['c'].append(tmp['c'])
    elif 'd' in tmp:
        data['d'].append(tmp['d'])
    elif 'e' in tmp:
        data['e'].append(tmp['e'])
    elif 'f' in tmp:
        data['f'].append(tmp['f'])
    elif 'g' in tmp:
        data['g'].append(tmp['g'])
    else:
        print("aisa kya guna kiya")

X_train = []
y_train = []

for category in data:
    if category == 'a':
        label = [1, 0, 0, 0, 0, 0, 0]
    elif category == 'b':
        label = [0, 1, 0, 0, 0, 0, 0]
    elif category == 'c':
        label = [0, 0, 1, 0, 0, 0, 0]
    elif category == 'd':
        label = [0, 0, 0, 1, 0, 0, 0]
    elif category == 'e':
        label = [0, 0, 0, 0, 1, 0, 0]
    elif category == 'f':
        label = [0, 0, 0, 0, 0, 1, 0]
    elif category == 'g':
        label = [0, 0, 0, 0, 0, 0, 1]

    label = np.array(label)

    for sample in data[category]:
        X_train.append(sample.reshape(2500 * 19,))
        y_train.append(label)

X_train = np.array(X_train)
y_train = np.array(y_train)

model = tf.keras.Sequential()
# Adds a densely-connected layer with 128 units to the model:
model.add(layers.Dense(128, activation=swish))
# Add another:
model.add(layers.Dense(64, activation=swish))
# Add a softmax layer with 10 output units:
model.add(layers.Dense(7, activation='softmax'))

model.compile(optimizer=tf.keras.optimizers.Adam(0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.fit(X_train, y_train, epochs=10)