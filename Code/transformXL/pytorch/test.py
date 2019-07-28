import torch

# demb = 10
# a = torch.arange(0.0, demb, 2.0)
# b = a / demb
# c = 10000 ** b
# inv_freq = 1 / c
# # inv_freq = 1 / (10000 ** (torch.arange(0.0, demb, 2.0) / demb))
# print(inv_freq)


a = torch.arange(0.0, 10, 2.0)
b = torch.arange(0.0, 10, 2.0)
c = a + b
print(c)