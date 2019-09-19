import keras
from keras.models import Model
from keras.layers import (Input, Dense, Conv2D, Flatten, GaussianNoise,
                          Concatenate, MaxPooling2D, Dropout, BatchNormalization)
import keras.backend as K

K.tensorflow_backend._get_available_gpus()

# This network consists of three parts:
#  * conv-nets on both eyes - directly detect which way they look
#  * conv-net on entire face - detect face/head orientation
#  * location/size of head
# All parts are at the end stiched together into a few dense layers

def get_model():
    left_input = Input((128, 128, 3))
    right_input = Input((128, 128, 3))

    # shared networks/weights between both eyes
    eye_layers = [
        GaussianNoise(0.5),
        Conv2D(96, (11, 11), strides=(4, 4), activation='relu'),
        BatchNormalization(),
        MaxPooling2D((3, 3), strides=2),
    #     Dropout(0.25),
        Conv2D(256, (5, 5), activation='relu', padding='same'),
        MaxPooling2D((3, 3), strides=2),
        Dropout(0.25),
        Conv2D(348, (3, 3), activation='relu', padding='same'),
        Dropout(0.25),
        Conv2D(64, (1, 1), activation='relu'),
        GaussianNoise(0.5),
        BatchNormalization() # +
    ]

    le = left_input
    for layer in eye_layers:
        le = layer(le)

    re = right_input
    for layer in eye_layers:
        re = layer(re)

    # combine both eyes into a dense layer
    eyes = Concatenate()([le, re])
    eyes = Dense(128, activation='relu')(eyes)
    eyes = BatchNormalization()(eyes)
    eyes = Flatten()(eyes)

    # identical (but separate) network for the face
    face_input = Input((128, 128, 3))

    f = face_input
    f = GaussianNoise(0.5)(f)
    f = Conv2D(96, (11, 11), strides=(4, 4), activation='relu')(f)
    f = BatchNormalization()(f)
    f = MaxPooling2D((3, 3), 2)(f)
    # f = Dropout(0.25)(f)
    f = Conv2D(256, (5, 5), activation='relu', padding='same')(f)
    f = MaxPooling2D((3, 3), 2)(f)
    f = Dropout(0.25)(f)
    f = Conv2D(348, (3, 3), activation='relu', padding='same')(f)
    f = Dropout(0.25)(f)
    f = Conv2D(128, (1, 1), activation='relu')(f)
    f = GaussianNoise(0.5)(f)
    f = BatchNormalization()(f) # +


    f = Dense(128, activation='relu')(f)
    f = BatchNormalization()(f)
    f = GaussianNoise(1)(f)
    # f = Dropout(0.25)(f)
    f = Dense(64, activation='relu')(f)
    f = BatchNormalization()(f)
    f = Flatten()(f)


    # TODO: try using a binary mask face grid instead
    data_input = Input((1+144, ))
    data = Dense(64, activation='relu')(data_input)
    data = BatchNormalization()(data)

    full = Concatenate()([eyes, f, data])

    full = Dense(128, activation='relu')(full)
    full = BatchNormalization()(full)
    full = GaussianNoise(0.5)(full)

    # full = Dense(48 * 27, activation='relu')(full)
    # full = Dropout(0.25)(full)
    # full = Dense(128, activation='relu')(full)
    # full = Dropout(0.5)(full)

    # A sigmoid activation function results 
    full = Dense(2, activation='sigmoid')(full)

    model = Model(inputs=[left_input, right_input, face_input, data_input], outputs=[full])


    opt = keras.optimizers.Adadelta(lr=1.0, rho=0.95, epsilon=None, decay=0.0)

    model.compile(loss='mean_squared_error',
                  optimizer=opt,
                 metrics=[])
    
    return model