# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np


#添加层
def layer_example(inputs, input_dim, output_dim, activation_function = None):
    Weights = tf.Variable(tf.random_normal([input_dim, output_dim]))
    print(Weights.shape)
    biases = tf.Variable(tf.zeros([1, output_dim]) + 0.1)

    W_plux_b = tf.matmul(inputs, Weights) + biases

    if activation_function is None:
        outputs = W_plux_b
    else:
        outputs = activation_function(W_plux_b)

    return outputs




#创建数据
#np.linspace(-1,1,300) 生成300个-1~1的数，然后加上维度
#[:,np.newaxis]    300*1
#[np.newaxis,:]    1*300
#x_data 300组数据，每组数据就一个1维向量
x_data = np.linspace(-1,1,300)[:,np.newaxis]
noise = np.random.normal(0, 0.05, x_data.shape)
y_data = np.square(x_data) - 0.5 + noise
print(x_data.shape)

xs = tf.placeholder(tf.float32, [None, 1])
ys = tf.placeholder(tf.float32, [None, 1])

#构建网络
input_layer_outputs = layer_example(xs, 1, 10, activation_function = tf.nn.relu)

predition_layer_outputs = layer_example(input_layer_outputs, 10, 1, activation_function=None)


loss = tf.reduce_mean(
            tf.reduce_sum(
                tf.square(ys - predition_layer_outputs), reduction_indices=[1]))

optimizer = tf.train.GradientDescentOptimizer(0.1)
train = optimizer.minimize(loss)

init = tf.global_variables_initializer()


with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
    sess.run(init)
    for step in range(1000):
        sess.run(train, feed_dict={xs:x_data, ys:y_data})
        if step % 50 == 0:
            print(sess.run(loss, feed_dict={xs:x_data, ys:y_data}))