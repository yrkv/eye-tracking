
import utils

#import keras
#from keras.callbacks import TensorBoard

import network

import numpy as np


(x_train, y_train), (x_test, y_test) = utils.load_data()

model = network.get_model()

model.fit(x_train, y_train / np.array([1920, 1080]),
                 validation_data=(x_test, y_test / np.array([1920, 1080])),
                 batch_size=64, epochs=100,
                 verbose=1)

# send model

print('send model')

model.save('/home/public/eye-tracking-models/model.h5')
