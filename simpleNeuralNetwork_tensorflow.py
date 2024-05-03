import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import keras
from keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from lab_coffee_utils import load_coffee_data, plt_roast
import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)
tf.autograph.set_verbosity(0)

# load data
X, Y = load_coffee_data();
print(X.shape, Y.shape)
plt_roast(X,Y)

# normalize data
norm_l = tf.keras.layers.Normalization(axis=-1)
norm_l.adapt(X)
Xn = norm_l(X)

# copy data to increase dataset
Xt = np.tile(Xn,(1000,1))
Yt = np.tile(Y,(1000,1))

# build the model
tf.random.set_seed(1234)
model = Sequential([
    tf.keras.Input(shape=(2,)), # expected shape of the input, can be use to explore but ignored in real case scenario
    Dense(3, activation='sigmoid', name='layer1'),
    Dense(1, activation='sigmoid', name='layer2')
])

model.summary() # print summary

# print w and b defined by tensor in the line 30
W1, b1 = model.get_layer("layer1").get_weights()
W2, b2 = model.get_layer("layer2").get_weights()
print(f"W1{W1.shape}:\n", W1, f"\nb1{b1.shape}:", b1)
print(f"W2{W2.shape}:\n", W2, f"\nb2{b2.shape}:", b2)

# define a loss function & specifies a compile opti
model.compile(
    loss = tf.keras.losses.BinaryCrossentropy(),
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.01),
)

# runs gradient descent and fits w to the data
# epochs = nb of training
model.fit(Xt, Yt, epochs=10)

# w has been updated after the training
W1, b1 = model.get_layer("layer1").get_weights()
W2, b2 = model.get_layer("layer2").get_weights()
print("W1:\n", W1, "\nb1:", b1)
print("W2:\n", W2, "\nb2:", b2)

# prediction test
X_test = np.array([
    [200,13.9],  # positive example
    [200,17]])   # negative example
X_testn = norm_l(X_test)
predictions = model.predict(X_testn)
print("predictions = \n", predictions)
yhat = (predictions >= 0.5).astype(int)
print(f"decisions = \n{yhat}")

