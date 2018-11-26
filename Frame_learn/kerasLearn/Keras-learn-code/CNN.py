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

    # data_format='channels_last'  指定输入shape = (samples, rows, cols, channels)
    # filters : 卷积核个数
    # kernel_size ： 卷积核shape，在不同channels上采用统一shape
    # strides : 步长
    # padding : 填充方式
    # kernel_initializer/bias_initializer : 初始化方法

    model.add(Conv2D(input_shape = (1,512,1), data_format='channels_last',
                     filters = 32, kernel_size = (1,5), strides = (1,1), padding = 'same',
                     kernel_initializer = 'RandomUniform', bias_initializer = 'Zeros'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(1,2), strides=(1,2), padding='valid', data_format='channels_last'))


    model.add(Conv2D(data_format='channels_last',
                     filters = 64, kernel_size = (1,5), strides = (1,1), padding = 'same',
                     kernel_initializer = 'RandomUniform', bias_initializer = 'Zeros'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(1,2), strides=(1,2), padding='valid', data_format='channels_last'))


    model.add(Conv2D(data_format='channels_last',
                     filters = 128, kernel_size = (1,3), strides = (1,1), padding = 'same',
                     kernel_initializer = 'RandomUniform', bias_initializer = 'Zeros'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(1,2), strides=(1,2), padding='valid', data_format='channels_last'))


    model.add(Conv2D(data_format='channels_last',
                     filters = 128, kernel_size = (1,3), strides = (1,1), padding = 'same',
                     kernel_initializer = 'RandomUniform', bias_initializer = 'Zeros'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(1,2), strides=(1,2), padding='valid', data_format='channels_last'))


    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(1024))
    model.add(Dense(512))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    model.summary()     # 输出各层信息
    model.compile(loss='binary_crossentropy',optimizer='rmsprop')
    model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(x_test, y_test))

    model.save(modelPath)


def predict(X_test, batch_size, modelPath, resultPath):
    my_model = load_model(modelPath)
    y_test = my_model.predict(X_test, batch_size=batch_size).tolist()

    file = open(resultPath, 'w+')
    for index in y_test:
        y = float(str(index).strip('\n').strip('\r').strip(' ').strip('[').strip(']'))
        file.write(str(y) + '\n')