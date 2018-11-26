# -*- coding:utf-8 -*-
import os
os.environ['KERAS_BACKEND']='tensorflow'

from keras.models import Sequential
from keras.layers.core import Dense,Dropout,Flatten
from keras.layers.core import Activation
from keras.layers.convolutional import Conv2D
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import MaxPooling2D
from keras.models import load_model
import codecs
import numpy as np

def generate_arrays(filePath, batch_size):

    srcFile = codecs.open(filename=filePath, mode='r', encoding='utf-8', errors='ignore')
    x_data_sum = []
    y_data_sum = []
    i = 0
    lines = srcFile.readlines()
    while 1:
        for line in lines:
            if line.strip('\n').strip('\r').strip(' ') == '':
                continue

            x_data = []
            s = line.strip('\n').strip('\r').strip(' ').split(' ')
            x = str(s[0])
            y = int(s[1])

            for char in x:
                try:
                    x_data.append(int(ord(char)))
                except:
                    print ('unexpected char' + ' : ' + char)
                    x_data.append(10)

            while len(x_data) < 1875:
                x_data.append(10)
            x_data = np.asarray(x_data)
            x_data = np.reshape(x_data, (25, 25, 3))
            x_data_sum.append(x_data)
            y_data_sum.append(y)
            i += 1
            if i == batch_size :
                x_data_sum = np.array(x_data_sum)
                y_data_sum = np.array(y_data_sum)
                yield (x_data_sum, y_data_sum)

                i = 0
                x_data_sum = []
                y_data_sum = []





def train(train_file_path, x_test, y_test, batch_size, epochs, modelPath):
    model=Sequential()

    # (25,25,3)
    model.add(Conv2D(input_shape=(25, 25, 3), data_format='channels_last',
                     filters=96, kernel_size=(1, 1), strides=(1, 1), padding='valid',
                     kernel_initializer='RandomUniform', bias_initializer='Zeros'))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='valid', data_format='channels_last'))

    # (12,12,3)
    model.add(Conv2D(data_format='channels_last',filters=256, kernel_size=(1, 1), strides=(1, 1), padding='same',
                     kernel_initializer='RandomUniform', bias_initializer='Zeros'))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid', data_format='channels_last'))

    # (6,6,3)
    model.add(Conv2D(data_format='channels_last', filters=384, kernel_size=(3, 3), strides=(1, 1), padding='same',
                     kernel_initializer='RandomUniform', bias_initializer='Zeros'))
    model.add(Activation('relu'))

    # (6,6,3)
    model.add(Conv2D(data_format='channels_last', filters=384, kernel_size=(3, 3), strides=(1, 1), padding='same',
                     kernel_initializer='RandomUniform', bias_initializer='Zeros'))
    model.add(Activation('relu'))

    # (6,6,3)
    model.add(Conv2D(data_format='channels_last', filters=256, kernel_size=(3, 3), strides=(1, 1), padding='same',
                     kernel_initializer='RandomUniform', bias_initializer='Zeros'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid', data_format='channels_last'))
    # (3,3,3)

    model.add(Flatten())

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

    # model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(x_test, y_test))

    model.fit_generator(generate_arrays(train_file_path, batch_size), steps_per_epoch = 19873, epochs=epochs, max_q_size=1000, validation_data=(x_test, y_test))

    model.save(modelPath)


def predict(X_test, batch_size, modelPath, resultPath):
    my_model = load_model(modelPath)
    y_test = my_model.predict(X_test, batch_size=batch_size).tolist()

    file = open(resultPath, 'w+')
    for index in y_test:
        y = float(str(index).strip('\n').strip('\r').strip(' ').strip('[').strip(']'))
        file.write(str(y) + '\n')