'''
@Project:MachineLearningAction

@Author:lincoln

@File:perceptron

@Time:2020-03-10 11:05:58

@Description:
'''
import numpy as np
#生成训练数据
x = np.array([[0.2, 0.5],[1.0, 1.0], [2.0, 1.0], [2.0, 2.0]])
y = np.array([[-1], [-1], [1], [1]])

#初始化参数
n_epochs = 30
lr = 0.1
w = np.random.randn(3, 1)

#添加偏置
x_bias = np.c_[np.ones((4, 1)), x]

#训练参数
for i in range(n_epochs):
    for j in range(len(x_bias)):
        p = y[j]*(x_bias[j].dot(w))
        if p <= 0:
            w = (w.reshape(1,3) + lr*(y[j]*x_bias[j])).reshape(3,1)
print(w)


