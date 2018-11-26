# -*- coding: utf-8 -*-
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

# torch.linspace(-1, 1, 100) 产生一个1维向量，包含100个元素，从-1~1均匀分布 [-1,...,1]
# torch.unsqueeze(x, dim=0/1) 为x增加一个维度  [[-1],...,[1]]
x = torch.unsqueeze(torch.linspace(-1, 1, 100), dim=1)  # x data (tensor), shape=(100, 1)
y = x.pow(2) + 0.2*torch.rand(x.size())                 # noisy y data (tensor), shape=(100, 1)

class Net(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super(Net, self).__init__()
        self.hidden = torch.nn.Linear(n_feature, n_hidden)   # hidden layer
        self.predict = torch.nn.Linear(n_hidden, n_output)   # output layer

    # 类似于Keras里的call()函数，定义网络的结构
    def forward(self, x):
        x = F.relu(self.hidden(x))      # activation function for hidden layer
        x = self.predict(x)             # linear output
        return x

net = Net(n_feature=1, n_hidden=10, n_output=1)     # define the network
print(net)  # net architecture

optimizer = torch.optim.SGD(net.parameters(), lr=0.2)
loss_func = torch.nn.MSELoss()  # this is for regression mean squared loss

plt.ion()   # something about plotting

for t in range(200):
    prediction = net(x)     # 等价于net.forward(x)，利用了class的__call__()函数的性质

    loss = loss_func(prediction, y)     # 必须是 (nn output, target)

    optimizer.zero_grad()   # torch中参数的梯度会累加，所以每次训练前都要清除之前的梯度记录
    loss.backward()         # 对和loss相关的、需要反向传播的参数计算梯度
    optimizer.step()        # 根据当前梯度和规则进行参数更新

    if t % 5 == 0:
        # plot and show learning process
        plt.cla()
        plt.scatter(x.data.numpy(), y.data.numpy())
        plt.plot(x.data.numpy(), prediction.data.numpy(), 'r-', lw=5)
        plt.text(0.5, 0, 'Loss=%.4f' % loss.data.numpy(), fontdict={'size': 20, 'color':  'red'})
        plt.pause(0.1)

plt.ioff()
plt.show()
