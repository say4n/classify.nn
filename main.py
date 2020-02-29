import utils
import tensorflow as tf
from tensorflow import keras
import tensorflow_addons as tfa
from tensorflow.keras import layers
import numpy as np
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt

SEED = 42

def swish(x):
    return x * tf.nn.sigmoid(x)

print("\n\n\t### train ###\n\n")

X_train, y_train, keys = utils.get_data("voweldata/*.mat")

model = tf.keras.Sequential()

model.add(layers.Dense(8192,
                       activation=swish,
                       kernel_initializer=keras.initializers.RandomNormal(seed=SEED)))
model.add(layers.Dense(1024,
                       activation=swish,
                       kernel_initializer=keras.initializers.RandomNormal(seed=SEED)))
model.add(layers.Dense(256,
                       activation=swish,
                       kernel_initializer=keras.initializers.RandomNormal(seed=SEED)))
model.add(layers.Dense(64,
                       activation=swish,
                       kernel_initializer=keras.initializers.RandomNormal(seed=SEED)))
model.add(layers.Dense(7,
                       activation='softmax',
                       kernel_initializer=keras.initializers.RandomNormal(seed=SEED)))

model.compile(optimizer=tf.keras.optimizers.Adam(1e-5),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.fit(X_train, y_train, epochs=5, verbose=1)

pred = list(map(lambda el: np.argmax(el), model.predict(X_train)))
actual = list(map(lambda el: np.argmax(el), y_train))

cm = tf.math.confusion_matrix(pred, actual, num_classes=len(keys))
df_cm = pd.DataFrame(cm.numpy(), index=keys, columns=keys)
plt.figure(figsize=(10,7))
sn.heatmap(df_cm, annot=True)
plt.show()