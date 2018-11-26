# -*- coding: utf-8 -*-


from keras.models import Sequential
from keras.layers import Dense
import matplotlib.pyplot as plt
import numpy as np
np.random.seed(1337)  # for reproducibility

# 创建数据
X = np.linspace(-1, 1, 200)
np.random.shuffle(X)    # randomize the data
Y = 0.5 * X + 2 + np.random.normal(0, 0.05, (200, ))
X_train, Y_train = X[:160], Y[:160]     # first 160 data points
X_test, Y_test = X[160:], Y[160:]       # last 40 data points

# 模型构建
model = Sequential()
# 模块第一层需要input_dim，后面的层不需要
# Dense()   =   activation(dot(input, kernel)+bias)
# 默认情况下 Dense() = dot(input, kernel)+bias        dot(a,b)是内积函数a*b
model.add(Dense(units=1, input_dim=1))

# choose loss function and optimizing method
model.compile(loss='mse', optimizer='sgd')

# training
print('Training -----------')
for step in range(301):
    cost = model.train_on_batch(X_train, Y_train)   # 每次都把所有训练数据代入训练
    if step % 100 == 0:
        print('train cost: ', cost)

# test
print('\nTesting ------------')
cost = model.evaluate(X_test, Y_test, batch_size=40)    # 测试效果如何
print('test cost:', cost)

# 获取模型第一层layers[0]的所有权重get_weights()
W, b = model.layers[0].get_weights()
print('Weights=', W, '\nbiases=', b)

# 测试集结果
Y_pred = model.predict(X_test)
plt.scatter(X_test, Y_test)
plt.plot(X_test, Y_pred)
plt.show()
