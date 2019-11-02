"""
Created on Sat Nov  2 20:09:18 2019

@author: yawning

在虹膜数据集上绘制决策树的决策面 => 决策树分类
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn import tree, datasets

# 参数和数据
n_classes = 3
plot_colors = "ryb" # 三原色?
plot_step = 0.02
iris = datasets.load_iris()

for idxpair, pair in enumerate([[0,1],[0,2],[0,3],[1,2],[1,3],[2,3]]):
    X = iris.data[:,pair] # pair[0], pair[1] 位置的特征
    y = iris.target
    clf = tree.DecisionTreeClassifier().fit(X, y)
    
    # 下面开始画图
    plt.subplot(2, 3, idxpair + 1)  # 开始画图啦
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1  #确立坐标系范围
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1 
    xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step),
                         np.arange(y_min, y_max, plot_step))  # 网格
    plt.tight_layout(h_pad=0.5, w_pad=0.5, pad=2.5)

    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    cs = plt.contourf(xx, yy, Z, cmap=plt.cm.RdYlBu)

    plt.xlabel(iris.feature_names[pair[0]])
    plt.ylabel(iris.feature_names[pair[1]])

    # 分好类, 可以染色
    for i, color in zip(range(n_classes), plot_colors):
        idx = np.where(y == i)
        plt.scatter(X[idx, 0], X[idx, 1], c=color, label=iris.target_names[i],
                    cmap=plt.cm.RdYlBu, edgecolor='black', s=15)

plt.suptitle("Decision surface of a decision tree using paired features")
plt.legend(loc='lower right', borderpad=0, handletextpad=0)
plt.axis("tight")

plt.figure()
clf = tree.DecisionTreeClassifier().fit(iris.data, iris.target)
tree.plot_tree(clf, filled=True)
plt.show()