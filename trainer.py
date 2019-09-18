
import keras
from keras.callbacks import TensorBoard

import utils

import network


(x_train, y_train), (x_test, y_test) = utils.load_data()

print(len(x_train))

model = network.get_model()

model.fit(x_train, y_train / np.array([1920, 1080]),
                 validation_data=(x_test, y_test / np.array([1920, 1080])),
                 batch_size=64, epochs=10,
                 verbose=0)

# send model

print('send model')
