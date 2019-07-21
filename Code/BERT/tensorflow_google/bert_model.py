# coding=utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import copy
import json
import math
import re
import numpy as np
import six
import tensorflow as tf
from basic_model_and_function import *
from embedding_model import *
from transformer_model import *

'''
疑问：
1. 当输入是句子A+B组成的输入时，怎么层输出的{T_i}里提取出B对应的输出
'''




class BertConfig(object):
    def __init__(self,
                 vocab_size,
                 hidden_size=768,
                 num_hidden_layers=12,
                 num_attention_heads=12,
                 intermediate_size=3072,
                 hidden_act="gelu",
                 hidden_dropout_prob=0.1,
                 attention_probs_dropout_prob=0.1,
                 max_position_embeddings=512,
                 type_vocab_size=16,
                 initializer_range=0.02):
        """Constructs BertConfig.
        Args:
            vocab_size: 词典大小
            hidden_size: Size of the encoder layers and the pooler layer.
            num_hidden_layers: Number of hidden layers in the Transformer encoder.
            num_attention_heads: Number of attention heads for each attention layer in
                the Transformer encoder.
            intermediate_size: The size of the "intermediate" (i.e., feed-forward)
                layer in the Transformer encoder.
            hidden_act: The non-linear activation function (function or string) in the
                encoder and pooler.
            hidden_dropout_prob: The dropout probability for all fully connected
                layers in the embeddings, encoder, and pooler.
            attention_probs_dropout_prob: The dropout ratio for the attention
                probabilities.
            max_position_embeddings: The maximum sequence length that this model might
                ever be used with. Typically set this to something large just in case
                (e.g., 512 or 1024 or 2048).
            type_vocab_size: The vocabulary size of the `token_type_ids` passed into
                `BertModel`.
            initializer_range: The stdev of the truncated_normal_initializer for
                initializing all weight matrices.
        """
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.initializer_range = initializer_range

    @classmethod
    def from_dict(cls, json_object):
        """Constructs a `BertConfig` from a Python dictionary of parameters."""
        config = BertConfig(vocab_size=None)
        for (key, value) in six.iteritems(json_object):
            config.__dict__[key] = value
        return config

    @classmethod
    def from_json_file(cls, json_file):
        """Constructs a `BertConfig` from a json file of parameters."""
        with tf.gfile.GFile(json_file, "r") as reader:
            text = reader.read()
        return cls.from_dict(json.loads(text))

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"

'''
BERT
参数：
input_ids : [batch_size, seq_length]，元素是每个词的序号
input_mask : [batch_size, seq_length]
token_type_ids : [batch_size, seq_length]
'''
class BertModel(object):

    def __init__(self,
                 config,
                 is_training,
                 input_ids,
                 input_mask=None,
                 token_type_ids=None,
                 use_one_hot_embeddings=False,
                 scope=None):
        """Constructor for BertModel.
        Args:
            config: `BertConfig` instance.
            is_training: bool. true for training model, false for eval model. Controls whether dropout will be applied.
            input_ids: int32 Tensor of shape [batch_size, seq_length].
            input_mask: (optional) int32 Tensor of shape [batch_size, seq_length].
            token_type_ids: (optional) int32 Tensor of shape [batch_size, seq_length].
            use_one_hot_embeddings: (optional) bool. Whether to use one-hot word
                embeddings or tf.embedding_lookup() for the word embeddings.
            scope: (optional) variable scope. Defaults to "bert".
        Raises:
            ValueError: The config is invalid or one of the input tensor shapes
                is invalid.
        """
        config = copy.deepcopy(config)
        # 测试时不使用dropout
        if not is_training:
            config.hidden_dropout_prob = 0.0
            config.attention_probs_dropout_prob = 0.0

        input_shape = get_shape_list(input_ids, expected_rank=2)
        batch_size = input_shape[0]
        seq_length = input_shape[1]

        # mark置1表示不处理
        if input_mask is None:
            input_mask = tf.ones(shape=[batch_size, seq_length], dtype=tf.int32)

        if token_type_ids is None:
            token_type_ids = tf.zeros(shape=[batch_size, seq_length], dtype=tf.int32)

        with tf.variable_scope(scope, default_name="bert"):
            with tf.variable_scope("embeddings"):
                # word embedding 和 对应词表
                (self.embedding_output, self.embedding_table) = embedding_lookup(
                        input_ids=input_ids,
                        vocab_size=config.vocab_size,
                        embedding_size=config.hidden_size,
                        initializer_range=config.initializer_range,
                        word_embedding_name="word_embeddings",
                        use_one_hot_embeddings=use_one_hot_embeddings)

                # Add positional embeddings and token type embeddings, then layer normalize and perform dropout.
                self.embedding_output = embedding_postprocessor(
                        input_tensor=self.embedding_output,
                        use_token_type=True,
                        token_type_ids=token_type_ids,
                        token_type_vocab_size=config.type_vocab_size,
                        token_type_embedding_name="token_type_embeddings",
                        use_position_embeddings=True,
                        position_embedding_name="position_embeddings",
                        initializer_range=config.initializer_range,
                        max_position_embeddings=config.max_position_embeddings,
                        dropout_prob=config.hidden_dropout_prob)

            with tf.variable_scope("encoder"):
                # mark.shape=[batch_size, seq_length] -> mark.shape=[batch_size, seq_length, seq_length]
                attention_mask = create_attention_mask_from_input_mask(input_ids, input_mask)

                # (batch_size, seq_length, embedding_size) -> (batch_size, seq_length, hidden_size)
                # embedding_size == hidden_size
                self.all_encoder_layers = transformer_model(
                        input_tensor=self.embedding_output,
                        attention_mask=attention_mask,
                        hidden_size=config.hidden_size,
                        num_hidden_layers=config.num_hidden_layers,
                        num_attention_heads=config.num_attention_heads,
                        intermediate_size=config.intermediate_size,
                        intermediate_act_fn=get_activation(config.hidden_act),
                        hidden_dropout_prob=config.hidden_dropout_prob,
                        attention_probs_dropout_prob=config.attention_probs_dropout_prob,
                        initializer_range=config.initializer_range,
                        do_return_all_layers=True)
            # 返回了所有blocks的输出，一般只用最后一个
            self.sequence_output = self.all_encoder_layers[-1]

            # 取出每个句子中第一个词对应的输出，并经过全连接层计算论文中的C
            with tf.variable_scope("pooler"):
                first_token_tensor = tf.squeeze(self.sequence_output[:, 0:1, :], axis=1)
                self.pooled_output = tf.layers.dense(
                        first_token_tensor,
                        config.hidden_size,
                        activation=tf.tanh,
                        kernel_initializer=create_initializer(config.initializer_range))

    def get_pooled_output(self):
        return self.pooled_output

    def get_sequence_output(self):
        return self.sequence_output

    def get_all_encoder_layers(self):
        return self.all_encoder_layers

    def get_embedding_output(self):
        return self.embedding_output

    def get_embedding_table(self):
        return self.embedding_table











