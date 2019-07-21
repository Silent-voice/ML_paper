
import collections
import copy
import json
import math
import re
import numpy as np
import six
import tensorflow as tf

from basic_model_and_function import *





def create_attention_mask_from_input_mask(from_tensor, to_mask):
    """Create 3D attention mask from a 2D tensor mask.
    Args:
        from_tensor: 2D or 3D Tensor of shape [batch_size, from_seq_length, ...].
        to_mask: int32 Tensor of shape [batch_size, to_seq_length].
    Returns:
        float Tensor of shape [batch_size, from_seq_length, to_seq_length].
    """
    from_shape = get_shape_list(from_tensor, expected_rank=[2, 3])
    batch_size = from_shape[0]
    from_seq_length = from_shape[1]

    to_shape = get_shape_list(to_mask, expected_rank=2)
    to_seq_length = to_shape[1]

    to_mask = tf.cast(tf.reshape(to_mask, [batch_size, 1, to_seq_length]), tf.float32)

    # We don't assume that `from_tensor` is a mask (although it could be). We
    # don't actually care if we attend *from* padding tokens (only *to* padding)
    # tokens so we create a tensor of all ones.
    #
    # `broadcast_ones` = [batch_size, from_seq_length, 1]
    broadcast_ones = tf.ones(shape=[batch_size, from_seq_length, 1], dtype=tf.float32)

    # Here we broadcast along two dimensions to create the mask.
    mask = broadcast_ones * to_mask

    return mask


'''
multi-head-attention
参数：
from_tensor、to_tensor : [batch_size, seq_length,width]，
                        from_tensor->Q，to_tensor->K、V，
                        self-attention中from_tensor=_tensor

attention_mask : [batch_size,from_seq_length, to_seq_length]
                mark字段，0会将其对应的attention scores设为负无穷，1不变

query_act、key_act、value_act : 计算Q、K、V时全连接层使用的激活函数

输出的特征维度：num_attention_heads * size_per_head

do_return_2d_tensor : 返回2维[batch_size*from_seq_length, num_attention_heads*size_per_head]还是3维
                    [batch_size, from_seq_length, num_attention_heads*size_per_head]

'''
def attention_layer(from_tensor,
                    to_tensor,
                    attention_mask=None,
                    num_attention_heads=1,
                    size_per_head=512,
                    query_act=None,
                    key_act=None,
                    value_act=None,
                    attention_probs_dropout_prob=0.0,
                    initializer_range=0.02,
                    do_return_2d_tensor=False,
                    batch_size=None,
                    from_seq_length=None,
                    to_seq_length=None):
    """
    Args:
        batch_size: (Optional) int. If the input is 2D, this might be the batch size
            of the 3D version of the `from_tensor` and `to_tensor`.
        from_seq_length: (Optional) If the input is 2D, this might be the seq length
            of the 3D version of the `from_tensor`.
        to_seq_length: (Optional) If the input is 2D, this might be the seq length
            of the 3D version of the `to_tensor`.
    Returns:
        float Tensor of shape [batch_size, from_seq_length,
            num_attention_heads * size_per_head]. (If `do_return_2d_tensor` is
            true, this will be of shape [batch_size * from_seq_length,
            num_attention_heads * size_per_head]).
    Raises:
        ValueError: Any of the arguments or tensor shapes are invalid.
    """

    # 对输入矩阵进行reshape和transpose
    def transpose_for_scores(input_tensor, batch_size, num_attention_heads, seq_length, width):
        output_tensor = tf.reshape(
                input_tensor, [batch_size, seq_length, num_attention_heads, width])
        # [batch_size, seq_length, num_attention_heads, width] -> [batch_size, num_attention_heads, seq_length, width]
        output_tensor = tf.transpose(output_tensor, [0, 2, 1, 3])
        return output_tensor

    from_shape = get_shape_list(from_tensor, expected_rank=[2, 3])
    to_shape = get_shape_list(to_tensor, expected_rank=[2, 3])

    if len(from_shape) != len(to_shape):
        raise ValueError(
                "The rank of `from_tensor` must match the rank of `to_tensor`.")

    if len(from_shape) == 3:
        batch_size = from_shape[0]
        from_seq_length = from_shape[1]
        to_seq_length = to_shape[1]
    elif len(from_shape) == 2:
        if (batch_size is None or from_seq_length is None or to_seq_length is None):
            raise ValueError(
                    "When passing in rank 2 tensors to attention_layer, the values "
                    "for `batch_size`, `from_seq_length`, and `to_seq_length` "
                    "must all be specified.")

    # Scalar dimensions referenced here:
    #     B = batch size (number of sequences)
    #     F = `from_tensor` sequence length
    #     T = `to_tensor` sequence length
    #     N = `num_attention_heads`
    #     H = `size_per_head`

    # 转出2维：(B*F, width)
    from_tensor_2d = reshape_to_matrix(from_tensor)
    to_tensor_2d = reshape_to_matrix(to_tensor)

    # (B*F, width) -> (B*F, N*H)
    query_layer = tf.layers.dense(
            from_tensor_2d,
            num_attention_heads * size_per_head,    # 输出维度
            activation=query_act,
            name="query",
            kernel_initializer=create_initializer(initializer_range))

    # (B*T, width) -> (B*T, N*H)
    key_layer = tf.layers.dense(
            to_tensor_2d,
            num_attention_heads * size_per_head,
            activation=key_act,
            name="key",
            kernel_initializer=create_initializer(initializer_range))

    # (B*T, width) -> (B*T, N*H)
    value_layer = tf.layers.dense(
            to_tensor_2d,
            num_attention_heads * size_per_head,
            activation=value_act,
            name="value",
            kernel_initializer=create_initializer(initializer_range))

    # (B*F, N*H) -> [B, N, F, H]
    query_layer = transpose_for_scores(query_layer, batch_size,
                                       num_attention_heads, from_seq_length,
                                       size_per_head)

    # (B*T, N*H) -> [B, N, T, H]
    key_layer = transpose_for_scores(key_layer, batch_size, num_attention_heads,
                                     to_seq_length, size_per_head)

    # 计算muli-head的attention_scores
    # `attention_scores` = [B, N, F, T]
    attention_scores = tf.matmul(query_layer, key_layer, transpose_b=True)
    attention_scores = tf.multiply(attention_scores,
                                   1.0 / math.sqrt(float(size_per_head)))

    # 与mark进行加权
    if attention_mask is not None:
        #变更维度 `attention_mask` = [B, 1, F, T]
        attention_mask = tf.expand_dims(attention_mask, axis=[1])
        # mark = 1 -> adder = 0.0; mark = 0 -> adder = -10000.0;
        adder = (1.0 - tf.cast(attention_mask, tf.float32)) * -10000.0
        # 更改attention_scores
        attention_scores += adder

    # Normalize the attention scores to probabilities.
    # `attention_probs` = [B, N, F, T]
    attention_probs = tf.nn.softmax(attention_scores)

    attention_probs = dropout(attention_probs, attention_probs_dropout_prob)

    # 对value进行reshape，以便后续与attention scores加权   (B*T, N*H) -> (B, T, N, H) -> (B, N, T, H)
    value_layer = tf.reshape(value_layer,[batch_size, to_seq_length, num_attention_heads, size_per_head])
    value_layer = tf.transpose(value_layer, [0, 2, 1, 3])

    # value与attention scores加权 = [B, N, F, H]
    context_layer = tf.matmul(attention_probs, value_layer)

    # reshape,准备输出 = [B, F, N, H]
    context_layer = tf.transpose(context_layer, [0, 2, 1, 3])

    if do_return_2d_tensor:
        # `context_layer` = [B*F, N*H]
        context_layer = tf.reshape(
                context_layer,
                [batch_size * from_seq_length, num_attention_heads * size_per_head])
    else:
        # `context_layer` = [B, F, N*H]
        context_layer = tf.reshape(
                context_layer,
                [batch_size, from_seq_length, num_attention_heads * size_per_head])

    return context_layer



'''
transformer : 只有encoder
参数：
input_tensor : [batch_size, seq_length, embedding_size]
attention_mask : Mask列表
hidden_size : Transformer的Attention层的输出：hidden_size = num_attention_heads * attention_head_size = embedding_size
num_hidden_layers : encoder里blocks(self-attention+...)的个数
intermediate_size、intermediate_act_fn : feed forward层的输出维度和激活函数
do_return_all_layers : 只输出最后一个blocks的输出，还是输出所有blocks的输出列表

输出维度：[batch_size, seq_length, hidden_size] 或者 [num_hidden_layers, batch_size, seq_length, hidden_size]


层结构：block = [input -> attention -> dense -> dropout -> norm -> feed_forward -> dropout -> norm ]
'''

def transformer_model(input_tensor,
                      attention_mask=None,
                      hidden_size=768,
                      num_hidden_layers=12,
                      num_attention_heads=12,
                      intermediate_size=3072,
                      intermediate_act_fn=gelu,
                      hidden_dropout_prob=0.1,
                      attention_probs_dropout_prob=0.1,
                      initializer_range=0.02,
                      do_return_all_layers=False):
    # hidden_size = num_attention_heads * attention_head_size
    if hidden_size % num_attention_heads != 0:
        raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (hidden_size, num_attention_heads))


    attention_head_size = int(hidden_size / num_attention_heads)
    input_shape = get_shape_list(input_tensor, expected_rank=3)
    batch_size = input_shape[0]
    seq_length = input_shape[1]
    input_width = input_shape[2]

    # embedding_size = hidden_size
    if input_width != hidden_size:
        raise ValueError("The width of the input tensor (%d) != hidden size (%d)" %
                                         (input_width, hidden_size))

    # 转为2维：[batch_size, seq_length, embedding_size] -> [batch_size*seq_length, embedding_size]
    prev_output = reshape_to_matrix(input_tensor)


    all_layer_outputs = []  # 收集每一个block的输出
    for layer_idx in range(num_hidden_layers):
        with tf.variable_scope("layer_%d" % layer_idx):
            layer_input = prev_output

            with tf.variable_scope("attention"):
                attention_heads = []
                with tf.variable_scope("self"):
                    attention_head = attention_layer(
                            from_tensor=layer_input,
                            to_tensor=layer_input,
                            attention_mask=attention_mask,
                            num_attention_heads=num_attention_heads,
                            size_per_head=attention_head_size,
                            attention_probs_dropout_prob=attention_probs_dropout_prob,
                            initializer_range=initializer_range,
                            do_return_2d_tensor=True,
                            batch_size=batch_size,
                            from_seq_length=seq_length,
                            to_seq_length=seq_length)
                    attention_heads.append(attention_head)

                attention_output = None
                if len(attention_heads) == 1:
                    attention_output = attention_heads[0]
                else:
                    # ???
                    # In the case where we have other sequences, we just concatenate
                    # them to the self-attention head before the projection.
                    attention_output = tf.concat(attention_heads, axis=-1)

                # Run a linear projection of `hidden_size` then add a residual
                # with `layer_input`.
                with tf.variable_scope("output"):
                    attention_output = tf.layers.dense(
                            attention_output,
                            hidden_size,
                            kernel_initializer=create_initializer(initializer_range))
                    attention_output = dropout(attention_output, hidden_dropout_prob)
                    attention_output = layer_norm(attention_output + layer_input)

            # The activation is only applied to the "intermediate" hidden layer.
            with tf.variable_scope("intermediate"):
                intermediate_output = tf.layers.dense(
                        attention_output,
                        intermediate_size,
                        activation=intermediate_act_fn,
                        kernel_initializer=create_initializer(initializer_range))

            # Down-project back to `hidden_size` then add the residual.
            with tf.variable_scope("output"):
                layer_output = tf.layers.dense(
                        intermediate_output,
                        hidden_size,
                        kernel_initializer=create_initializer(initializer_range))
                layer_output = dropout(layer_output, hidden_dropout_prob)
                layer_output = layer_norm(layer_output + attention_output)
                prev_output = layer_output
                all_layer_outputs.append(layer_output)

    # 处理返回结果
    if do_return_all_layers:
        final_outputs = []
        for layer_output in all_layer_outputs:
            final_output = reshape_from_matrix(layer_output, input_shape)
            final_outputs.append(final_output)
        return final_outputs
    else:
        final_output = reshape_from_matrix(prev_output, input_shape)
        return final_output