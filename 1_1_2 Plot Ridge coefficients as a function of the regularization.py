# -*- coding: utf-8 -*-
"""
Created on Tue Oct 22 21:36:17 2019

@author: yawning
绘制不同正则化情况下的相关系数
"""
import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model

# 建立数据和超参数
X = 1. / (np.arange(1, 11) + np.arange(0, 10)[:, np.newaxis]) #10x10 Hilbert 矩阵
y = np.ones(10)
n_alpha = 200  # 超参数个数
alphas = np.logspace(-10, -2, n_alpha) 

# 建立模型
coefs = []
for a in alphas:
    reg = linear_model.Ridge(alpha = a,fit_intercept = False)  # 不计算截断项
    reg.fit(X,y)
    coefs.append(reg.coef_)

# 画图展示结果
ax = plt.gca()
ax.plot(alphas,coefs)
ax.set_xscale('log')  #x改为log均分
ax.set_xlim(ax.get_xlim()[::-1])  # 逆转 x
plt.xlabel('alpha')
plt.ylabel('weights')
plt.title('Ridge coefficients as a function of the regularization')
plt.axis('tight')  #让x轴和y轴单位长度相等，即分辨率相等
plt.show()