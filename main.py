import numpy as np
import random
from tensorflow import keras as k
import matplotlib.pyplot as plt

import tensorflow as tf
(x_train, y_train), (x_test, y_test) = k.datasets.mnist.load_data()

x_train = x_train
x_test = x_test



model = k.Sequential([
                      k.layers.Flatten(input_shape=(28, 28)),
                      k.layers.Dense(256, 'relu'),
                      k.layers.Dense(10, "softmax")
])


model.compile(k.optimizers.SGD(learning_rate=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()

model.fit(x_train, y_train, epochs=60)
test_loss, test_acc = model.evaluate(x_test, y_test)

print('Test_acc = ', test_acc, "\n")
print('Test_loss = ', test_loss, '\n')








