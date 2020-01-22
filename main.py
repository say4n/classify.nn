import utils
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def swish(x):
    return x * tf.nn.sigmoid(x)

print("\n\n\t### train ###\n\n")

X_train, y_train = utils.get_data("voweldata/*.mat")

model = tf.keras.Sequential()

model.add(layers.Dense(8192, activation=swish))
model.add(layers.Dense(1024, activation=swish))
model.add(layers.Dense(256, activation=swish))
model.add(layers.Dense(64, activation=swish))
model.add(layers.Dense(7, activation='softmax'))

model.compile(optimizer=tf.keras.optimizers.Adam(1e-4),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.fit(X_train, y_train, epochs=15, verbose=1)

print("\n\n\t### evaluate ### \n\n")

X_test, y_test = utils.get_data("testdata/*.mat")
model.evaluate(X_test, y_test, verbose=2)