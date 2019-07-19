from attention_model import *
from E_D_model import *
from embedding_model import *
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math, copy, time
from torch.autograd import Variable
import matplotlib.pyplot as plt
import seaborn      #绘图库
seaborn.set_context(context="talk")




'''
参数说明：
src_vocab : source词典大小
tgt_vocab : target词典大小
N : Encoder和Decoder的层数
d_model : embedding维度
d_ff : feed_forward的输出维度
h : head个数
'''

def make_model(src_vocab, tgt_vocab, N=6, d_model=512, d_ff=2048, h=8, dropout=0.1):

    c = copy.deepcopy

    # Multi-head attention
    attn = MultiHeadedAttention(h, d_model)
    # feed_forward
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)

    position = PositionalEncoding(d_model, dropout)

    model = EncoderDecoder(
        Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
        Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout), N),
        nn.Sequential(Embeddings(d_model, src_vocab), c(position)),
        nn.Sequential(Embeddings(d_model, tgt_vocab), c(position)),
        Generator(d_model, tgt_vocab))

    # 初始化参数
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform(p)
    return model