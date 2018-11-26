# -*- coding: utf-8 -*-
import tensorflow as tf

x = tf.placeholder(tf.float32,shape=[None,75],name='x')
y_ = tf.placeholder(tf.int32,shape=[None,],name='y_')