# -*- coding: utf-8 -*-
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

tf.set_random_seed(1)   # set random seed

# 导入数据
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

# 设置模型参数
lr = 0.001                  # learning rate
training_iters = 100000     # train step 上限
batch_size = 128            # 批处理大小
n_inputs = 28               # MNIST data input (img shape: 28*28)
n_steps = 28                # time steps
n_hidden_units = 128        # neurons in hidden layer
n_classes = 10              # MNIST classes (0-9 digits)

# 每组数据 28*28，每次输入是1*28，输入28次
# 每批数据 (128*28*28)

# x y placeholder
x = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
y = tf.placeholder(tf.float32, [None, n_classes])

# 对 weights biases 初始值的定义
weights = {
    # shape (28, 128)
    'in': tf.Variable(tf.random_normal([n_inputs, n_hidden_units])),
    # shape (128, 10)
    'out': tf.Variable(tf.random_normal([n_hidden_units, n_classes]))
}
biases = {
    # shape (128, )
    'in': tf.Variable(tf.constant(0.1, shape=[n_hidden_units, ])),
    # shape (10, )
    'out': tf.Variable(tf.constant(0.1, shape=[n_classes, ]))
}


# 嵌入层 => LSTM层 => 全连接层
def RNN(X, weights, biases):
    ########################################
    # 嵌入层：(128,28,28) => (128,28,128)

    # shape=(128,28,28) => (128*28,28)
    X = tf.reshape(X, [-1, n_inputs])
    # (128*28,28) * (28*128) = (128*28，128)
    X_in = tf.matmul(X, weights['in']) + biases['in']
    # (128*28，128) => (128,28,128)
    X_in = tf.reshape(X_in, [-1, n_steps, n_hidden_units])

    ##########################################
    # LSTN层
    # RNN cell对每个输入都会产生一个状态信息state，一般是副状态h_state
    # LSTM cell则会产生两个state,主状态c_state和副状态h_state

    # n_hidden_units 输入的神经元个数     state_is_tuple=True 输出的状态是一个元组(c_state, h_state)
    cell = tf.contrib.rnn.BasicLSTMCell(n_hidden_units, forget_bias=1.0, state_is_tuple=True)
    # 初始化状态，28个输入，每个输入都有一个状态
    init_state = cell.zero_state(batch_size, dtype=tf.float32)

    # time_major=False (batch, steps, inputs)   time_major=True (steps, batch, inputs) as X_in.
    # final_state = (c_state, h_state) 两个最终的状态,状态的shape=(128)
    # outputs shape = (128,28,128)
    outputs, final_state = tf.nn.dynamic_rnn(cell, X_in, initial_state=init_state, time_major=False)


    #############################################
    # 对输出加一个全连接层
    # results = tf.matmul(final_state[1], weights['out']) + biases['out']

    # # or
    # tf.transpose() 转换不同维度的数据，第二个参数是转换后的维度
    # (batch, steps, inputs) = (0,1,2) => (1,0,2) = (steps, batch, inputs)
    # tf.unstack()是将tensor变量转为list变量，这里就变成steps个元素，每个元素是一个(batch, inputs)
    outputs = tf.unstack(tf.transpose(outputs, [1,0,2]))

    # 这时候outputs的每一个元素就是这一批数据(128)在某个输入上产生的所有状态
    # outputs[-1] == final_state[1]     这一批数据在最后一个输入上产生的状态
    results = tf.matmul(outputs[-1], weights['out']) + biases['out']    # shape = (128, 10)

    return results


# 预测结果
pred = RNN(x, weights, biases)
# 损失函数
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
# 优化器
train_op = tf.train.AdamOptimizer(lr).minimize(cost)

# 评价指标
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))


# 启动会话
with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    step = 0
    while step * batch_size < training_iters:
        # 提取一批训练数据
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        batch_xs = batch_xs.reshape([batch_size, n_steps, n_inputs])
        # 传入数据，训练模型
        sess.run([train_op], feed_dict={
            x: batch_xs,
            y: batch_ys,
        })

        # 输出训练结果
        if step % 20 == 0:
            print(sess.run(accuracy, feed_dict={
            x: batch_xs,
            y: batch_ys,
            }))
        step += 1