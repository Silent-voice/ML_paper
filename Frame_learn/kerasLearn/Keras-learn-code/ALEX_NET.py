# -*- coding:utf-8 -*-
import os
os.environ['KERAS_BACKEND']='tensorflow'

from keras.models import Sequential
from keras.layers.core import Dense,Dropout,Flatten
from keras.layers.core import Activation
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.models import load_model


def train(x_train, y_train, x_test, y_test, batch_size, epochs, modelPath):
    model=Sequential()

    model.add(Conv2D(input_shape=(227, 227, 3), data_format='channels_last',
                     filters=96, kernel_size=(11, 11), strides=(4, 4), padding='valid',
                     kernel_initializer='RandomUniform', bias_initializer='Zeros'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='valid', data_format='channels_last'))


    model.add(Conv2D(data_format='channels_last',filters=256, kernel_size=(5, 5), strides=(1, 1), padding='same',
                     kernel_initializer='RandomUniform', bias_initializer='Zeros'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='valid', data_format='channels_last'))


    model.add(Conv2D(data_format='channels_last', filters=384, kernel_size=(3, 3), strides=(1, 1), padding='same',
                     kernel_initializer='RandomUniform', bias_initializer='Zeros'))
    model.add(Activation('relu'))


    model.add(Conv2D(data_format='channels_last', filters=384, kernel_size=(3, 3), strides=(1, 1), padding='same',
                     kernel_initializer='RandomUniform', bias_initializer='Zeros'))
    model.add(Activation('relu'))


    model.add(Conv2D(data_format='channels_last', filters=256, kernel_size=(3, 3), strides=(1, 1), padding='same',
                     kernel_initializer='RandomUniform', bias_initializer='Zeros'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='valid', data_format='channels_last'))


    model.add(Dense(4096))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Dense(4096))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    model.summary()  # 输出各层信息
    model.compile(loss='binary_crossentropy', optimizer='rmsprop')
    model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(x_test, y_test))

    model.save(modelPath)


def predict(X_test, batch_size, modelPath, resultPath):
    my_model = load_model(modelPath)
    y_test = my_model.predict(X_test, batch_size=batch_size).tolist()

    file = open(resultPath, 'w+')
    for index in y_test:
        y = float(str(index).strip('\n').strip('\r').strip(' ').strip('[').strip(']'))
        file.write(str(y) + '\n')