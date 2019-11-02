"""
1.10 Decision Trees
    练习决策树模型 tree
    https://scikit-learn.org/stable/modules/tree.html
"""
from sklearn import tree
import numpy as np
# import graphviz
from sklearn.datasets import load_iris  # 虹膜
'''1.10.1 分类  Classification
DecisionTreeClassifier(criterion=’gini’, splitter=’best’, max_depth=None, 
    min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, 
    max_features=None, random_state=None, max_leaf_nodes=None, min_impurity_decrease=0.0, 
    min_impurity_split=None, class_weight=None, presort=False)
    criterion: 如何对样本进行划分. 可选: 'gini' (the Gini impurity) 
               'entropy' (the information gain) 信息增益
    splitter: 当前选择哪个属性进行划分.  可选: 'best' / 'random'
    max_depth: 树的最大深度. None =>将节点展开至所有叶子都是纯净的, or 叶子节点数 少于最小节点
    min_samples_split: 拆分内部节点所需的最少样本数
    min_samples_leaf： 在叶节点处所需的最小样本数 (没看懂???)
    min_weight_fraction_leaf: 在所有叶节点处的权重总和中的最小加权分数 (还是没看懂???)
    max_features: 寻找最佳分割时要考虑的功能数量:int/float/'auto'/'sqrt'/'log2'/None
    max_leaf_nodes: 以该数字为最好的方式种植tree, None 即不考虑
    min_impurity_decrease: 如果节点分裂导致杂质减少大于或等于该值，则该节点将分裂
    min_impurity_split: 如果节点的杂质高于阈值，它将分裂
    class_weight: 与类有关的权重
    presort: 是否对数据进行预排序以加快寻找最佳拟合的速度
'''
iris = load_iris()
X1, y1 = iris.data, iris.target
clf1 = tree.DecisionTreeClassifier(random_state = 0, criterion = 'entropy', max_depth =3)
clf1.fit(X1[:-1],y1[:-1])  # 拟合
print(clf1.apply([X1[-1]]))  # 返回样本被预测为的叶子的索引
print(clf1.decision_path([X1[-1]])) # 返回样本在树中的决策路径
print(clf1.feature_importances_) # 返回每个属性的重要程度
print(clf1.get_depth(), clf1.get_n_leaves()) # 返回树的深度, 叶子个数
print(clf1.predict([X1[-1]]), clf1.predict_proba([X1[-1]]))  # 预测样本类型, 以及每个类型可能的概率
tree.plot_tree(clf1)
r = tree.export.export_text(clf1, feature_names=iris['feature_names'])
print(r)  # 画树












