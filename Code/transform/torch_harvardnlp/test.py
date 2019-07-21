import torch
import math

max_len = 10
d_model = 5
pe = torch.zeros(max_len, d_model)
position = torch.arange(0., max_len).unsqueeze(1)
div_term = torch.exp(torch.arange(0., d_model, 2) *
                -(math.log(10000.0) / d_model))
pe[:, 0::2] = torch.sin(position * div_term)
# pe[:, 1::2] = torch.cos(position * div_term)

x = pe[:, 0::2]
print(x)
# y = pe[:, 1::2]