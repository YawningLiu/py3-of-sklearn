# -*- coding: utf-8 -*-
"""
Created on Tue Oct 22 20:43:36 2019

@author: yawning
"""
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score

# 建立数据集
diabetes = datasets.load_diabetes()  # 关于糖尿病的数据
X = diabetes.data[:,np.newaxis,2]  # 2 表示第2个位置的数 =>变为 1 维情况
X_train,X_test = X[:-20], X[-20:]
y_train,y_test = diabetes.target[:-20], diabetes.target[-20:]

# 建立模型
reg = linear_model.LinearRegression()
reg.fit(X_train,y_train)
y_pred = reg.predict(X_test)

# 相关系数和结果
print('Intercept and coefficients: \n', reg.intercept_, reg.coef_)
print("均方误差: %.2f" % mean_squared_error(y_test, y_pred))
# Explained variance score: 1 is perfect prediction
print('r2误差: %.2f'% r2_score(y_test, y_pred))  # 和下面的等价
# print('r2误差: %.2f'% reg.score(X_test,y_test))

# 作图
plt.scatter(X_test, y_test,  color = 'black')
plt.plot(X_test, y_pred, color = 'red', linewidth = 3)
plt.xticks(()) #删去刻度?
plt.yticks(()) #同上
plt.show()  # 显示图像
