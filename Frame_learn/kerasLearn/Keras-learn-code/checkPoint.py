# -*- coding: utf-8 -*-

'''
使用ModelCheckpoint函数每训练一轮，比较一下效果是否有提高，提高的话就保存一次模型

keras.callbacks.ModelCheckpoint(filepath,monitor='val_loss',verbose=0,save_best_only=False, save_weights_only=False, mode='auto', period=1) 

参数说明：
    filename：字符串，保存模型的路径
    monitor：需要监视的值，评估指标
    verbose：信息展示模式，0或1(checkpoint的保存信息，类似Epoch 00001: saving model to ...)
    save_best_only：当设置为True时，监测值有改进时才会保存当前的模型
    mode：‘auto’，‘min’，‘max’之一，在save_best_only=True时决定性能最佳模型的评判准则，例如，当监测值为val_acc时，模式应为max，当监测值为val_loss时，模式应为min。在auto模式下，评价准则由被监测值的名字自动推断。
    save_weights_only：若设置为True，则只保存模型权重，否则将保存整个模型（包括模型结构，配置信息等）
    period：CheckPoint之间的间隔的epoch数
'''

import os
os.environ['KERAS_BACKEND']='tensorflow'

from keras.models import Sequential
from keras.layers.core import Dense,Dropout,Flatten
from keras.layers.core import Activation
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.callbacks import ModelCheckpoint


from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Activation, Convolution2D, MaxPooling2D, Flatten
from keras.optimizers import Adam
import numpy as np
np.random.seed(1337)  # for reproducibility

# 创建数据
(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = X_train.reshape(-1, 1, 28, 28)/255.
X_test = X_test.reshape(-1, 1, 28, 28)/255.
y_train = np_utils.to_categorical(y_train, num_classes=10)
y_test = np_utils.to_categorical(y_test, num_classes=10)

# 模型构建
model = Sequential()
model.add(Convolution2D(batch_input_shape=(None, 1, 28, 28),filters=32,kernel_size=5,strides=1,padding='same',data_format='channels_first',))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=2,strides=2,padding='same',data_format='channels_first',))
model.add(Convolution2D(filters=64, kernel_size=5, strides=1, padding='same', data_format='channels_first'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=2, strides=2, padding='same', data_format='channels_first'))
model.add(Flatten())
model.add(Dense(1024))
model.add(Activation('relu'))
model.add(Dense(10))
model.add(Activation('softmax'))

# 自定义优化器
adam = Adam(lr=1e-4)
model.compile(optimizer=adam,loss='categorical_crossentropy',metrics=['accuracy'])

# 保存中间模型
temp_model_path = "./save/model_{epoch:02d}-{val_acc:.2f}.hdf5"
checkpoint = ModelCheckpoint(filepath=temp_model_path, monitor='val_acc', verbose=1,save_best_only=True)

print('Training ------------')
model.fit(X_train, y_train, epochs=100, batch_size=64)
model.save('./save/model_final.hdf5')


