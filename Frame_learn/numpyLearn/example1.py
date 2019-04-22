# -*- coding: utf-8 -*-
import numpy as np

# 直接进行运算操作是针对每个元素
x1 = np.random.rand(10,3)
x2 = x1**2
# print(x1)
# print(x2)

# 对第3行元素赋值
x2[2, :] = [1,2,3]
# print(x2)


minValue = x1.min()     # 所有元素最小值
minValue = x1.min(axis=0)   # 每列最小值
minValue = x1.min(axis=1)   # 每行最小值

# 与min()类似
x1.sum()
x1.sum(axis=0)
x1.sum(axis=1)

# argsort() 从小到大排序，返回排序后的相应下标
x3 =x1.sum(axis=1)
x4 = x3.argsort()
# print(x3)
# print(x4)


d = [1,2,3]
D = np.tile(d, (3,2))   # 生成一个3*6的数组，，每行由2个d组成
# print(D)