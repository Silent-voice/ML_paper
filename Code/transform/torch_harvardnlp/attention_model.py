from basic_model import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import math





'''
Scaled Dot Product Attention

X.masked_fill(mask, value) : 
    1. mask是一个列表，将mask中值为1的位置对应的x置位value
    2. mask[i] == 1 -> X[i] = value
'''
def attention(query, key, value, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) \
             / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim = -1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn



'''
Multi-head attention
1. 每个head只学习元素数据中的部分维度
'''
class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        super(MultiHeadedAttention, self).__init__()
        # d_model = h * attention_out_size
        assert d_model % h == 0

        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        "Implements Figure 2"

        # 对mask再加1维，和q/k/v维度保持一致
        if mask is not None:
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        '''
        zip(A,B) => [(a_1,b_1),...,(a_n,b_n)]
        X.view(shape) : 类似于reshape()
        
        原始输入维度：(batch, element_num, embedding_size)
        经过全连接层然后切分后维度：(nbatches, element_num, h, embedding_size/h)
        '''
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]

        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = attention(query, key, value, mask=mask,
                                 dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        '''
        将后两个维度再次拼接在一起
        '''
        x = x.transpose(1, 2).contiguous() \
            .view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)