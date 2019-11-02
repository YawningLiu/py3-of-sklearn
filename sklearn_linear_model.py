'''
1.1 Generalized Linear Model
    练习广义线性回归模型
    w_i:系数 为 coef_
    w_0: 截断项 为 intercept_
    主要学习linear_model
    LinearRegression / Ridge / Lasso / ElasticNet
    https://scikit-learn.org/stable/modules/linear_model.html#
'''
import numpy as np
from sklearn import linear_model 
from sklearn.datasets import make_regression

'''1.1.1 linear_model.LinearRegression 最小二乘法
LinearRegression(fit_intercept=True, normalize=False, 
                 copy_X=True, n_jobs=None)
    fit_intercept: boolean,可选,默认True. 是否计算截断项
    normalize: boolean,可选. 是否对回归变量进行归一化
    copy_X: boolean,可选. 是否复制X. (否则会覆盖原X)
    n_jobs: None/int, 可选. 用于计算的样本数. 
输出: coef_, intercept_
'''
reg1 = linear_model.LinearRegression()
X1 = [[-3, 0], [1, 1], [2, 7]]
y1 = [0, 1, 5]
reg1.fit(X1, y1)   # 拟合
print('-'*15 + '1.1.1' + '-'*15)
print(reg1.coef_, reg1.intercept_)  # 系数和截断项
print(reg1.get_params())  # 获取模型构造时输入的参数
print(reg1.predict([[4, 4]])) # 预测新值, 注意输入要是二维矩阵
print(reg1.score(X1,y1))  # 预测决定系数R^2

'''1.1.2 Ridge Regression => 含有正则项的最小二乘法 (正则项为L2范数)
Ridge(alpha=1.0, fit_intercept=True, normalize=False, copy_X=True, 
      max_iter=None, tol=0.001, solver='auto', random_state=None)
    alpha: 正则数, 必须为正浮点数. 正则数越大正则化越强 
    max_iter:  梯度法的最大迭代步数. (solver为迭代法时)
    tol: 答案的精度误差. (solver为迭代法时)
    solver: {'auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga’}
            求解问题时具体使用的方法. (sag/saga: 随机平均梯度下降)
    random_state: 伪随机数生成器, 别的没看懂. 
输出: coef_, intercept_, n_iter_ (非梯度法为None) 
'''
reg2 = linear_model.Ridge(alpha=0.4)  # 正则项超参数为0.5 
X2 = [[-3, 0], [1, 1], [2, 7]]
y2 = [0, 1, 5]
reg2.fit(X2,y2)
print('-'*15 + '1.1.2' + '-'*15)
print(reg2.coef_, reg2.intercept_)  # 系数和截断项
print(reg2.get_params())  # 获取模型构造时输入的参数
print(reg2.predict([[4, 4]])) # 预测新值, 注意输入要是二维矩阵
print(reg2.score(X2,y2))  # 预测决定系数R^2
'''RidgeCV 广义交叉验证 => 留一法 计算超参数alpha
RidgeCV(alphas=(0.1, 1.0, 10.0), fit_intercept=True, normalize=False, 
        scoring=None, cv=None, gcv_mode=None, store_cv_values=False)
    alphas: 正浮点数组 => CV就是找最优值
    scoring: 选超参数的判断标准, 一般默认R^2
    cv: 交叉验证方法, 默认为留一法
    gcv_mode: 没看懂,大概是对不同超参数计算时使用的具体方法?
    store_cv_values: 是否应将与每个alpha对应的交叉验证值存储在cv_values_中.
输出: coef_, intercept_, alpha_, cv_values_(if store_cv_values=True and cv=None, 
    将包含所有alpha的均方误差)
'''
reg22 = linear_model.RidgeCV(alphas=np.logspace(-6, 6, 13))
reg22.fit(X2,y2)   
print(">>> 1.1.2.3:")
print(reg22.alpha_,reg2.coef_, reg2.intercept_)  # 系数和截断项, 超参数
print(reg22.score(X2,y2))  # 预测决定系数R^2

'''1.1.3 LASSO => 正则项为L1范数的最小二乘法, 用于稀疏系数矩阵(p252)
Lasso(alpha=1.0, fit_intercept=True, normalize=False, precompute=False, 
      copy_X=True, max_iter=1000, tol=0.0001, warm_start=False, 
      positive=False, random_state=None, selection=’cyclic’)
    precompute: 是否使用预先计算的Gram矩阵来加快计算速度
    warm_start: 是否使用上一个fit的结果作初始值
    positive: 是否强制系数为正
    selection: 'random' 会使收敛加快
输出: coef_, intercept_, n_iter_, sparse_coef_(以稀疏矩阵的格式输出)
'''
reg3 = linear_model.Lasso(alpha=0.4)  # 正则项超参数为0.5 
X3 = [[-3, 0], [1, 1], [2, 7]]
y3 = [0, 1, 5]
reg3.fit(X3,y3)
print('-'*15 + '1.1.3' + '-'*15)
print(reg3.coef_, reg3.intercept_)  # 系数和截断项
print(reg3.sparse_coef_)  # 稀疏形式
print(reg3.predict([[4, 4]])) # 预测新值, 注意输入要是二维矩阵
print(reg3.score(X2,y2))  # 预测决定系数R^2
'''交叉验证:LassoCV - 同上,基本相同
LassoLarsCV(基于最小角) - 可以自己计算出最优alpha, 但是数据很多时会很慢. 
''' 

'''1.1.4 Multi-task Lasso => 误差项是F范数, 正则项是l1l2范数
相当于把多个y并列一起求, 
MultiTaskLasso(alpha=1.0, fit_intercept=True, normalize=False, copy_X=True, 
     max_iter=1000, tol=0.0001, warm_start=False, random_state=None, selection=’cyclic’)
参数与上面相同,详情见例子
'''

'''1.1.5. Elastic-Net => 正则项为 L1范数 + L2 范数的结合 
ElasticNet(alpha=1.0, copy_X=True, fit_intercept=True, l1_ratio=0.5,
      max_iter=1000, normalize=False, positive=False, precompute=False,
      random_state=0, selection='cyclic', tol=0.0001, warm_start=False)

'''
X, y = make_regression(n_features=2, random_state=0)
reg5 = linear_model.ElasticNet(random_state=0)
reg5.fit(X, y)  
print('-'*15 + '1.1.5' + '-'*15)
print(reg5.coef_, reg5.intercept_) 
print(reg5.predict([[0, 0]])) 

'''1.1.8. LARS Lasso => 使用最小角回归的Lasso
LassoLars(alpha=1.0, fit_intercept=True, verbose=False, normalize=True, 
      precompute=’auto’, max_iter=500, eps=2.220446049250313e-16, copy_X=True, 
      fit_path=True, positive=False)
'''  
reg8 = linear_model.LassoLars(alpha=0.1)
X8 = [[-3, 0], [1, 1], [2, 7]]
y8 = [0, 1, 5]
reg8.fit(X8,y8)
print('-'*15 + '1.1.8' + '-'*15)
print(reg8.coef_, reg8.intercept_)  # 系数和截断项
print(reg8.predict([[4, 4]])) # 预测新值, 注意输入要是二维矩阵




