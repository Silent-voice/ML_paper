# -*- coding: utf-8 -*-
import numpy as np

# 参数意思分别 是从a 中以概率P，随机选择3个, p没有指定的时候相当于是一致的分布


# memory = np.random.rand(10, 2 * 2 + 2)
# print(memory)
#
# a1 = np.random.choice(a=5, size=3, replace=False, p=None)
# print(a1)
#
# print(memory[a1, :])

batch_index = np.arange(32, dtype=np.int32)
print(batch_index)