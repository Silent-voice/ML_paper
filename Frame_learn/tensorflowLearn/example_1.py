# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np


### 初始化数据 ###
x_data = np.random.rand(100).astype(np.float32)     #100个float32类型的数据
y_data = x_data*0.1 + 0.3


### 构建模型 ###

#random_normal(shape, mean=0.0, stddev=1.0, dtype=tf.float32, seed=None, name=None)    从正态分布输出随机值
#random_uniform(shape, minval=0, maxval=None, dtype=tf.float32, seed=None, name=None)   从均匀分布中返回随机值

Weights = tf.Variable(tf.random_uniform([1],-1.0,1.0))  #Weights是一个变量，shape = 1 ,大小在-1.0~1.0
biases = tf.Variable(tf.zeros([1]))     #基准值biases

y = Weights * x_data + biases       #预测值

#损失函数
#tf.square  计算平方
#tf.reduce_mean 求tensor平均值              reduce_max 求最大值
loss = tf.reduce_mean(tf.square(y - y_data))


#优化器，负责最小化损失
#GradientDescentOptimizer(learning_rate)   实现梯度下降算法的优化器,learning_rate是学习率
optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(loss)

init = tf.global_variables_initializer()    #初始化变量



### 开始训练 ###
sess = tf.Session()     #创建会话，只有在会话中才可以运行上面建立的模型
sess.run(init)


#训练200次，每次都用所有数据进行训练
for step in range(200):
    sess.run(train)
    #每训练20次，打印一下当前的权重
    if step % 20 == 0:
        print(step, sess.run(Weights), sess.run(biases))

sess.close()





#placeholder    一种Variable，不同之处在于他的值是运行时传入的，而不是模型内部生成的.
#通过 run() 里的 feed_dict={} 传值
input1 = tf.placeholder(tf.float32)
input2 = tf.placeholder(tf.float32)

output = tf.multiply(input1, input2)

with tf.Session() as sess:
    print( sess.run(output, feed_dict={input1 : [7.], input2 : [2.]}) )


