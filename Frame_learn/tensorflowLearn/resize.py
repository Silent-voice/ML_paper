# -*- coding: utf-8 -*-

from skimage import transform,data
import matplotlib.pyplot as plt
img = data.camera()
dst=transform.resize(img, (256, 256))

print (str(type(img[0][0])))
print (str(img))
print (str(dst))



'''
output = transform.resize(img, output_shape)
img : 两个维度
output_shape : 两个元素的元组

img ：中每个元素的类型是uint8，即8位int.   np.uint8()
output ：输出的是除以(2^8-1)即255后的数据

'''