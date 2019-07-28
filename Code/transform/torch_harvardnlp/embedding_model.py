import torch
import torch.nn as nn
from torch.autograd import Variable
import math

class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        # vocab：整个字典的大小，和keras类似
        # d_model：嵌入向量大小，每个元素最终生成的向量维度
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)

'''
位置编码
1. 直接使用三角函数进行计算
2. 位置编码对于确定的维度而言是固定的
'''
class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        '''
        pe.shape=(max_len, d_model)     初始全0
        position.shape=(max_len,1)      值为绝对位置1,2,...
        div_term.shape=(d_model/2)      一组很小的数
        position * div_term = (max_len,d_model/2)
        '''
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0., max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0., d_model, 2) *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)       # 0::2 表示下标0,2,4,8,...的值，0是起始位置，2是间隔
        pe[:, 1::2] = torch.cos(position * div_term)       # 1::2 表示下标1,3,5,7,...的值，1是起始位置，2是间隔
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)  # 不优化

    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)],
                         requires_grad=False)
        return self.dropout(x)

