# -*- coding: utf-8 -*-
"""
Created on Thu Oct 31 16:12:57 2019

@author: yawning
Joint feature selection with multi-task Lasso
合并多个回归问题, 从而使所选特征在各个任务中保持相同, 从而更稳定 
"""
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import MultiTaskLasso, Lasso

rng = np.random.RandomState(42)  # 保证随机得到的值相同

# 构造数据
n_samples, n_features, n_tasks = 100, 30, 40
n_relevant_features = 5 # 相关的特征只有5个
coef = np.zeros((n_tasks, n_features))
time = np.linspace(0, 2*np.pi, n_tasks)
for k in range(n_relevant_features):  # 构造相关系数
    coef[:,k] = np.sin((1. + rng.randn(1)) * time  + 3 * rng.randn(1))
    
X = rng.randn(n_samples, n_features)
Y = np.dot(X,coef.T) + rng.randn(n_samples, n_tasks)   # .T 转置

coef_lasso = np.array([Lasso(alpha=0.5).fit(X, y).coef_ for y in Y.T]) 
coef_multitasklasso = MultiTaskLasso(alpha=1.).fit(X, Y).coef_

# 画图 => 抄
# Plot support and time series
fig = plt.figure(figsize=(8, 5))
plt.subplot(1, 2, 1)
plt.spy(coef_lasso)
plt.xlabel('Feature')
plt.ylabel('Time (or Task)')
plt.text(10, 5, 'Lasso')
plt.subplot(1, 2, 2)
plt.spy(coef_multitasklasso)
plt.xlabel('Feature')
plt.ylabel('Time (or Task)')
plt.text(10, 5, 'MultiTaskLasso')
fig.suptitle('Coefficient non-zero location')

feature_to_plot = 3  # 0-4 画哪个特征值
plt.figure()
lw = 2
plt.plot(coef[:, feature_to_plot], color='seagreen', linewidth=lw,
         label='Ground truth')
plt.plot(coef_lasso[:, feature_to_plot], color='cornflowerblue', linewidth=lw,
         label='Lasso')
plt.plot(coef_multitasklasso[:, feature_to_plot], color='gold', linewidth=lw,
         label='MultiTaskLasso')
plt.legend(loc='upper center')
plt.axis('tight')
plt.ylim([-1.1, 1.1])
plt.show()


