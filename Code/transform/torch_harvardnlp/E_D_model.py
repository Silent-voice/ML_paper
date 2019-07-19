from basic_model import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


'''
FeedForward
'''
class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))



'''
EncoderLayer : self_attn + feed_forward + 残差网络
数据流 : X -> norm -> self_attn -> dropout -> add -> new_X -> norm -> feed_forward -> dropout -> add -> Z
'''
class EncoderLayer(nn.Module):
    "Encoder is made up of self-attn and feed forward (defined below)"
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        # sublayer : [SublayerConnection, SublayerConnection] 两个残差层组成的列表
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        # 使用第一个残差层，数据流： (X,X,X) -> norm -> self_attn -> dropout -> add -> tem_O
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        # 使用第二个残差层，数据流： tem_O -> norm -> feed_forward -> dropout -> add -> Z
        return self.sublayer[1](x, self.feed_forward)



'''
编码器Encoder : N * EncoderLayer
数据流：X -> EncoderLayer -> ... -> EncoderLayer -> Z
'''
class Encoder(nn.Module):
    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask):
        "Pass the input (and mask) through each layer in turn."
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


'''
DecoderLayer : masked_self_attn + src_attn + feed_forward + 残差网络
数据流 : (X,X,X) -> norm -> self_attn -> dropout -> add -> Q + Z -> (Q,Z,Z) -> norm -> self_attn -> dropout -> add -> tem_O -> norm -> feed_forward -> dropout -> add -> Y

masked_self_attn : 
    1. 第一个DecoderLayer的第一个masked_self_attn输入的是预测向量Y
    2. Y的元素一开始是全被masked，之后每计算出一个y_i，相应masked就会去掉
'''
class DecoderLayer(nn.Module):
    "Decoder is made of self-attn, src-attn, and feed forward (defined below)"

    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 3)

    def forward(self, x, memory, src_mask, tgt_mask):
        m = memory
        # 使用第一个残差层，数据流： (X,X,X) -> norm -> self_attn -> dropout -> add -> Q
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        # 使用第二个残差层，数据流： (Q,Z,Z) -> norm -> self_attn -> dropout -> add -> tem_O
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
        # 使用第三个残差层，数据流： tem_O -> norm -> feed_forward -> dropout -> add -> Y
        return self.sublayer[2](x, self.feed_forward)



'''
解码器Decoder : N * DecoderLayer
数据流：Z -> DecoderLayer -> ... -> DecoderLayer -> Y
'''
class Decoder(nn.Module):
    "Generic N layer decoder with masking."

    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)



'''
encoder : (x_1,...,x_n)->(z_1,...,z_n)
decoder : 
1. (z_1,...,z_n)->(y_1,...,y_n)
2. 每一次计算生成一个y_i
3. 自循环，上次生成的y_i作为下次计算y_i+1的额外输入

'''

class EncoderDecoder(nn.Module):

    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator

    def forward(self, src, tgt, src_mask, tgt_mask):
        "Take in and process masked src and target sequences."
        return self.decode(self.encode(src, src_mask), src_mask,
                           tgt, tgt_mask)

    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)

    def decode(self, memory, src_mask, tgt, tgt_mask):
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)

