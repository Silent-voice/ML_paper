import torch
import torch.nn as nn
import torch.nn.functional as F
import copy


'''
用于复制生成相同的模型
'''
def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


'''
Generator : 用于生成 Linear + softmax层
'''
class Generator(nn.Module):
    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        # d_model : 输入维度        vocab : 输出维度
        self.proj = nn.Linear(d_model, vocab)

    def forward(self, x):
        return F.log_softmax(self.proj(x), dim=-1)



'''
归一化层
'''
class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


'''
残差网络：x -> norm -> layer -> dropout -> add -> new_x
'''
class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    # CustomLayer : 自定义层
    def forward(self, x, CustomLayer):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(CustomLayer(self.norm(x)))