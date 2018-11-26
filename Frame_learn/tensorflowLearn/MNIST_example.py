# -*- coding: utf-8 -*-
import input_data
import tensorflow as tf

# 卷积 -> 池化 -> 卷积 -> 池化 -> 全连接 -> Dropout



# 用来初始化Weight
def weight_variable(shape):
    # truncated_normal 生成正太分布的数据
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

# tf.nn.conv2d(input, filter, strides, padding, use_cudnn_on_gpu=None, name=None)
# 第一个参数input：指需要做卷积的输入图像，它要求是一个Tensor，具有[batch, in_height, in_width, in_channels]这样的shape，具体含义是[训练时一个batch的图片数量, 图片高度, 图片宽度, 图像通道数]，注意这是一个4维的Tensor，要求类型为float32和float64其中之一
# 第二个参数filter：相当于CNN中的卷积核，它要求是一个Tensor，具有[filter_height, filter_width, in_channels, out_channels]这样的shape，具体含义是[卷积核的高度，卷积核的宽度，图像通道数，卷积核个数]，要求类型与参数input相同，有一个地方需要注意，第三维in_channels，就是参数input的第四维
# 第三个参数strides：卷积时在图像每一维的步长，这是一个一维的向量，长度4
# 第四个参数padding：string类型的量，只能是"SAME","VALID"其中之一，这个值决定了不同的卷积方式
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

# tf.nn.max_pool(value, ksize, strides, padding, name=None)
# value：池化的输入，一般池化层接在卷积层的后面，所以输出通常为feature map。feature map依旧是[batch, in_height, in_width, in_channels]这样的参数。
# ksize：池化窗口的大小，参数为四维向量，通常取[1, height, width, 1]，因为我们不想在batch和channels上做池化，所以这两个维度设为了1。ps：估计面tf.nn.conv2d中stries的四个取值也有              相同的意思。
# stries：步长，同样是一个四维向量。
# padding：填充方式同样只有两种不重复了。

# in_height, in_width 都缩小一倍
def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')

# 创建数据
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
x_input = tf.placeholder("float", shape=[None, 784])
y_input = tf.placeholder("float", shape=[None, 10])

x_image = tf.reshape(x_input, [-1,28,28,1])

# 第一层卷积层与池化层
W_conv1 = weight_variable([5, 5, 1, 32])    # 卷积的每个特征平面是5*5,1个通道,32个特征平面
b_conv1 = bias_variable([32])

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)    # 输出 28*28*32
h_pool1 = max_pool_2x2(h_conv1)                             # 输出 14*14*32

# 第二层卷积层与池化层
W_conv2 = weight_variable([5, 5, 32, 64])   # 卷积的每个特征平面是5*5,32个通道,64个特征平面
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)    # 输出 14*14*64
h_pool2 = max_pool_2x2(h_conv2)                             # 输出 7*7*64

# 现在输出是7*7


# 将4维输出转为2维
h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])


# 非线性全连接层1，激活函数relu
W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

# Dropout
keep_prob = tf.placeholder("float")
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)


# 非线性全连接层2，激活函数softmax
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])
y_predict = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

loss = -tf.reduce_sum(y_input*tf.log(y_predict))

train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)


# 评测性能
correct_prediction = tf.equal(tf.argmax(y_predict,1), tf.argmax(y_input,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))


init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)


for i in range(20000):
    batch = mnist.train.next_batch(50)
    if i%100 == 0:
        train_accuracy =  sess.run(accuracy, feed_dict={x_input: batch[0], y_input: batch[1], keep_prob: 1.0})
        print ("step %d, training accuracy %g"%(i, train_accuracy))

    sess.run(train_step, feed_dict={x_input: batch[0], y_input: batch[1], keep_prob: 0.5})

# 输出测试结果
batch = mnist.train.next_batch(50)
# 这里求y_predict时不需要用y_input
result = sess.run(y_predict, feed_dict={x_input: batch[0], keep_prob: 0.5})

print ("test accuracy %g"%sess.run(accuracy, feed_dict={x_input: batch[0], y_input: batch[1], keep_prob: 1.0}))

sess.close()