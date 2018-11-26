# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np

def Save():
    # 训练数据
    x_data = np.random.rand(100).astype(np.float32)     #100个float32类型的数据
    x_data = np.reshape(x_data, [-1,1])
    # print(x_data.shape)
    y_data = x_data*0.1 + 0.3

    x_input = tf.placeholder(tf.float32, shape=[None, 1])
    y_input = tf.placeholder(tf.float32, shape=[None, 1])

    Weights = tf.Variable(tf.random_uniform([1,1],-1.0,1.0))  #Weights是一个变量，shape = (1,10) ,大小在-1.0~1.0


    biases = tf.Variable(tf.zeros([1,1]))     #基准值biases

    y = tf.matmul(x_input,Weights) + biases       #预测值

    loss = tf.reduce_mean(tf.square(y - y_input))
    optimizer = tf.train.GradientDescentOptimizer(0.5)
    train = optimizer.minimize(loss)

    init = tf.global_variables_initializer()    #初始化变量

    sess = tf.Session()
    sess.run(init)


    #训练200次，每次都用所有数据进行训练
    for step in range(200):
        sess.run(train, feed_dict = {x_input:x_data, y_input:y_data})

    # 保存模型
    # 参数是指明需要保存的变量，默认
    saver = tf.train.Saver(tf.global_variables())
    # saver = tf.train.Saver()




    # tf.train.Saver.save(sess, save_path, global_step=None, latest_filename=None, meta_graph_suffix='meta', write_meta_graph=True)
    # sess:  用于保存变量的Session
    # save_path:  checkpoint 文件的路径。如果saver 是共享的，这是共享checkpoint 文件名的前缀。
    # global_step:  如果提供了global step number，将会追加到 save_path 后面去创建checkpoint 的文件名。可选参数可以是一个Tensor，一个name Tensor或integer Tensor.
    
    # eg:
    # saver.save(sess, 'my-model', global_step=0) ==>filename: 'my-model-0'  
    # saver.save(sess, 'my-model', global_step=1000) ==>filename: 'my-model-1000' 
    saver.save(sess, './save/model.ckpt')

    sess.close()



    # 定期保存模型的代码样例
    # 每次保存都是一个检查点checkpoint，模型可以从任一次检查点恢复
    # for step in xrange(1000000):  
    # sess.run(...training_op...)  
    # if step % 1000 ==0:  
        # saver.save(sess,'my-model',global_step=step)

# 加载时也需要把原模型整个运算都写出来，不过不需要重新训练了
def Reload():
    x_input = tf.placeholder("float", shape=[None, 1])
    y_input = tf.placeholder("float", shape=[None, 1])

    Weights = tf.Variable(tf.random_uniform([1], -1.0, 1.0))  # Weights是一个变量，shape = 1 ,大小在-1.0~1.0
    biases = tf.Variable(tf.zeros([1]))  # 基准值biases

    y = tf.matmul(Weights, x_input) + biases  # 预测值

    loss = tf.reduce_mean(tf.square(y - y_input))
    optimizer = tf.train.GradientDescentOptimizer(0.5)
    train = optimizer.minimize(loss)


    saver = tf.train.Saver()
    sess = tf.Session()
    # tf.train.Saver.restore(sess, save_path)
    saver.restore(sess, './save/model.ckpt')

    # 获取原model参数
    print(sess.run(Weights))
    print(sess.run(biases))


    # 测试
    x_test_data = np.random.rand(100).astype(np.float32)
    y_predict = sess.run(y, feed_dict= {x_input : x_test_data})

    sess.close()


Save()