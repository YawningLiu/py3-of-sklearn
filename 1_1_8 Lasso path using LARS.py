# -*- coding: utf-8 -*-
"""
Created on Fri Nov  1 20:08:57 2019

@author: yawning
Computing regularization path using the LARS
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn import datasets

# 导入数据
diabetes = datasets.load_diabetes()
X = diabetes.data
y = diabetes.target
_, _, coefs = linear_model.lars_path(X, y, method='lasso', verbose=True)
xx = np.sum(np.abs(coefs.T), axis=1)  # 按列相加 
xx /= xx[-1]

# 画图
plt.plot(xx,coefs.T)
ymin,ymax = plt.ylim()
plt.vlines(xx, ymin, ymax, linestyle='dashed')
plt.xlabel('|coef| / max|coef|')
plt.ylabel('Coefficients')
plt.title('LASSO Path')
plt.axis('tight')
plt.show()