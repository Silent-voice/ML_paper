# -*- coding:utf-8 -*-
import os
os.environ['KERAS_BACKEND']='tensorflow'
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers.core import Dense
from keras.layers.core import Dropout
from keras.layers.core import Activation
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.models import load_model
import numpy as np

def train(max_features, x_train, y_train, x_test, y_test, batch_size, epochs, modelPath):

    #创建Sequential序贯模型
    model=Sequential()

    #添加网络层

    #嵌入层：用来将每个正整数下标转为固定大小的向量
    #max_features 是最大下标+1
    #128 代表转化后的向量大小为128，即每个向量有128个数组成
    model.add(Embedding(max_features,128,input_length=75))

    model.add(LSTM(128))
    model.add(Dropout(0.5))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    #编译模型
    #loss 是损失函数
    #optimizer 是优化器
    model.compile(loss='binary_crossentropy',optimizer='rmsprop')

    #对初始的二维数据进行处理，不足75位的填0，多的截断
    x_train=sequence.pad_sequences(x_train,maxlen=75)
    x_test = sequence.pad_sequences(x_test, maxlen=75)

    # Train where y_train is 0-1
    #训练模型
    #batch_size 是每次迭代时训练的样本个数
    #nb_epoch 训练轮数
    # model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=1)
    model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(x_test, y_test))


    # 模型保存
    # 文件 .h5     sudo pip install h5py
    model.save(modelPath)



def predict(X_test, batch_size, modelPath, resultPath):
    X_test = sequence.pad_sequences(X_test, maxlen=75)
    my_model = load_model(modelPath)
    y_test = my_model.predict(X_test, batch_size=batch_size).tolist()

    file = open(resultPath, 'w+')
    for index in y_test:
        y = float(str(index).strip('\n').strip('\r').strip(' ').strip('[').strip(']'))
        file.write(str(y) + '\n')





