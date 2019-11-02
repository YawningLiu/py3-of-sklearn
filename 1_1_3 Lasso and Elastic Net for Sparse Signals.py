# -*- coding: utf-8 -*-
"""
Created on Thu Oct 24 16:11:19 2019

@author: yawning
用手动生成的添加噪声的稀疏信号估计Lasso和Elastic-Net回归模型
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn import linear_model

###################################################################
# 生成稀疏信号
np.random.seed(42)  # 可以使多次生成的随机数相同

n_samples, n_features = 50, 100
X = np.random.randn(n_samples,n_features) # 生成d1 *d2 矩阵
 
idx = np.arange(n_features) # 等价于range(n_features)
coef = (-1) ** idx *np.exp(-idx /10 ) # n_features 个相关系数
coef[10:] = 0  # 将系数变得稀疏些
y = np.dot(X,coef)  

# 添加噪声, 以高斯概率分布产生随机数, 戒断整数后,等价于 np.random.randn(size= n_samples)
y += 0.01 * np.random.normal(size = n_samples)  

# 分割数据集为训练集和测试集
n_samples = X.shape[0]  # 如果不是自己构造的数据, 是不知道大小的的
X_train, y_train = X[:n_samples // 2], y[:n_samples // 2]
X_test, y_test =  X[n_samples // 2:], y[n_samples // 2:]

###################################################################
# lasso
alpha = 0.1
lasso = linear_model.Lasso(alpha=alpha)
lasso.fit(X_train, y_train)  # 训练
y_pred_lasso = lasso.predict(X_test)
r2_score_lasso = r2_score(y_test,y_pred_lasso)  # 使用lasso自带score又predict一遍
print(lasso)
print("r^2 on test data : %f" % r2_score_lasso)

###################################################################
# ElasticNet
enet = linear_model.ElasticNet(alpha=alpha, l1_ratio=0.7)
enet.fit(X_train, y_train)  # 训练
y_pred_enet = enet.predict(X_test)
r2_score_enet = r2_score(y_test,y_pred_enet)  # 使用lasso自带score又predict一遍
print(enet)
print("r^2 on test data : %f" % r2_score_enet)

###################################################################
#plot => stem 棉棒图主要用来显示一个包含正负的数据集

# 画 Elastic net 系数非0项, 绿色
m, s, _ = plt.stem(np.where(enet.coef_)[0], enet.coef_[enet.coef_ != 0],
                   markerfmt='x', label='Elastic net coefficients')
plt.setp([m, s], color="#2ca02c")
# 画 LAsso 系数非0 项, 橙色
m, s, _ = plt.stem(np.where(lasso.coef_)[0], lasso.coef_[lasso.coef_ != 0],
                   markerfmt='x', label='Lasso coefficients')
plt.setp([m, s], color='#ff7f0e')
# 画真实系数         
plt.stem(np.where(coef)[0], coef[coef != 0], label='true coefficients',
         markerfmt='bx')

plt.legend(loc='best')
plt.title("Lasso $R^2$: %.3f, Elastic Net $R^2$: %.3f"
          % (r2_score_lasso, r2_score_enet))
plt.show()

