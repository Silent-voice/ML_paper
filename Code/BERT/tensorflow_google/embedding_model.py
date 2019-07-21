import collections
import copy
import json
import math
import re
import numpy as np
import six
import tensorflow as tf

from basic_model_and_function import *

'''
论文中提及的是：Token Embedding + Segment Embedding + Position Embedding
代码里却是：Word Embedding + Token Embedding(默认不使用，应该就是Segment Embedding) + Position Embedding
'''




'''
Word Embedding : 随机初始化一个Embedding词表，然后将词对应的Embedding取出，这个词表是一个Variable，会在在训练时会更新
参数：
input_ids : [batch_size, seq_length]，seq_length是句子长度，这里的元素id指的是词对应的序号
vocab_size : 词典大小
use_one_hot_embeddings : 从词典中提取Embedding的方式
输出：[batch_size, seq_length, embedding_size]
'''
def embedding_lookup(input_ids,
                     vocab_size,
                     embedding_size=128,
                     initializer_range=0.02,
                     word_embedding_name="word_embeddings",
                     use_one_hot_embeddings=False):

    # This function assumes that the input is of shape [batch_size, seq_length, num_inputs].
    # If the input is a 2D tensor of shape [batch_size, seq_length], we reshape to [batch_size, seq_length, 1].
    if input_ids.shape.ndims == 2:
        input_ids = tf.expand_dims(input_ids, axis=[-1])

    # 生成一个随机的词表，shape=(vocab_size, embedding_size)
    embedding_table = tf.get_variable(
            name=word_embedding_name,
            shape=[vocab_size, embedding_size],
            initializer=create_initializer(initializer_range))

    # 拉平操作，转为1维向量
    flat_input_ids = tf.reshape(input_ids, [-1])
    if use_one_hot_embeddings:
        # 将id转为对应的独热码
        one_hot_input_ids = tf.one_hot(flat_input_ids, depth=vocab_size)
        # 将独热码转为词表中对应的embedding向量
        output = tf.matmul(one_hot_input_ids, embedding_table)
    else:
        # 和one-hot方式结果类似
        output = tf.gather(embedding_table, flat_input_ids)


    # 调整维度：input.shape=(batch_size,seq_length,num_inputs) -> output.shape=(batch_size,seq_length,num_inputs*embedding_size)
    input_shape = get_shape_list(input_ids)
    output = tf.reshape(output, input_shape[0:-1] + [input_shape[-1] * embedding_size])
    return (output, embedding_table)


'''
在Word Embedding的基础上再加上Position Embedding和Token Embedding
参数：
input_tensor：[batch_size, seq_length, embedding_size]

token_type_ids：[batch_size, seq_length] 每个词对应的token类型
token_type_vocab_size：token类型的总个数

max_position_embeddings：句子长度的最大值，不能小于输入中句子的长度

输出：[batch_size, seq_length, embedding_size]，与输入维度相同
'''
def embedding_postprocessor(input_tensor,
                            use_token_type=False,
                            token_type_ids=None,
                            token_type_vocab_size=16,
                            token_type_embedding_name="token_type_embeddings",
                            use_position_embeddings=True,
                            position_embedding_name="position_embeddings",
                            initializer_range=0.02,
                            max_position_embeddings=512,
                            dropout_prob=0.1):

    input_shape = get_shape_list(input_tensor, expected_rank=3)
    batch_size = input_shape[0]
    seq_length = input_shape[1]
    width = input_shape[2]

    output = input_tensor

    '''
    1. 和计算Word Embedding一样，先随机初始化一个token embedding词表[token_type_vocab_size, width]，之后在训练时更新
    2. 根据每个词对应的token类型从词表中提取相应embedding
    3. 将token_type_embeddings与原来的embedding直接相加
    '''
    if use_token_type:
        if token_type_ids is None:
            raise ValueError("`token_type_ids` must be specified if"
                             "`use_token_type` is True.")
        token_type_table = tf.get_variable(
                name=token_type_embedding_name,
                shape=[token_type_vocab_size, width],
                initializer=create_initializer(initializer_range))
        # This vocab will be small so we always do one-hot here, since it is always
        # faster for a small vocabulary.
        flat_token_type_ids = tf.reshape(token_type_ids, [-1])
        one_hot_ids = tf.one_hot(flat_token_type_ids, depth=token_type_vocab_size)
        token_type_embeddings = tf.matmul(one_hot_ids, token_type_table)
        token_type_embeddings = tf.reshape(token_type_embeddings,[batch_size, seq_length, width])
        output += token_type_embeddings


    '''
    1. 随机初始化一个full_position_embeddings词表，该词表中的position个数是大于句子中position的个数的
    2. 从full_position_embeddings词表中抽取seq_length个embedding组成position_embeddings
    3. position embedding对于所有数据而言都是相同的，所以所有数据都加上一个相同的position embedding
    '''
    if use_position_embeddings:
        # 确保max_position_embeddings >= seq_length
        assert_op = tf.assert_less_equal(seq_length, max_position_embeddings)
        with tf.control_dependencies([assert_op]):
            full_position_embeddings = tf.get_variable(
                    name=position_embedding_name,
                    shape=[max_position_embeddings, width],
                    initializer=create_initializer(initializer_range))

            # full_position_embeddings.shape = (max_position_embeddings, width)
            # position_embeddings.shape = (seq_length, width)
            # 从full_position_embeddings的第一个维度中抽取seq_length个组成position_embeddings
            position_embeddings = tf.slice(full_position_embeddings, [0, 0],[seq_length, -1])
            num_dims = len(output.shape.as_list())

            # output的第一维是batch size，但对于position embedding而言，每个数据的位置编码都是相同的，
            # 所以不需要第一维，所有数据都加上相同的position embedding
            position_broadcast_shape = []
            for _ in range(num_dims - 2):
                position_broadcast_shape.append(1)
            position_broadcast_shape.extend([seq_length, width])
            position_embeddings = tf.reshape(position_embeddings,position_broadcast_shape)
            output += position_embeddings

    output = layer_norm_and_dropout(output, dropout_prob)
    return output