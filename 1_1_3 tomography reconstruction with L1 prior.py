# -*- coding: utf-8 -*-
"""
Created on Thu Oct 24 21:01:20 2019

@author: yawning
压缩感知: 用Lasso进行CT重建 - 从一组沿不同角度获取的平行投影重建图像
"""
import numpy as np
from scipy  import sparse
from scipy import ndimage  # 用于图像处理. ndimage表示一个n维图像
import matplotlib.pyplot as plt
from sklearn import linear_model 

def _weights(x, dx=1, orig=0):  # 没按懂想干啥
    x = np.ravel(x)   #降为1维
    floor_x = np.floor((x - orig) / dx).astype(np.int64)
    alpha = (x - orig - floor_x * dx) / dx
    return np.hstack((floor_x, floor_x + 1)), np.hstack((1 - alpha, alpha))

def _generate_center_coordinates(l_x):  # 生成坐标中心
    X, Y = np.mgrid[:l_x, :l_x].astype(np.float64) # 相当于找到了所有整数点的坐标
    center = l_x / 2.
    X += 0.5 -center
    Y += 0.5 -center
    return X,Y
    
def build_projection_operator(l_x, n_dir):  # 计算层析设计矩阵
    # l_x: 矩阵大小, n_dir: 投影的角度数
    # 得到 (n_dir l_x, l_x**2) 形状的稀疏矩阵
    X, Y = _generate_center_coordinates(l_x)
    angles = np.linspace(0, np.pi, n_dir, endpoint=False)
    data_inds, weights, camera_inds = [], [], []
    data_unravel_indices = np.arange(l_x ** 2)
    data_unravel_indices = np.hstack((data_unravel_indices,
                                      data_unravel_indices))  # 在水平方向上平铺
    for i, angle in enumerate(angles):  # 对于所有给定角度
        Xrot = np.cos(angle) * X - np.sin(angle) * Y
        inds, w = _weights(Xrot, dx=1, orig=X.min())
        mask = np.logical_and(inds >= 0, inds < l_x)  # 逻辑与 => 且
        weights += list(w[mask])
        camera_inds += list(inds[mask] + i * l_x)
        data_inds += list(data_unravel_indices[mask])
    proj_operator = sparse.coo_matrix((weights, (camera_inds, data_inds)))
    return proj_operator

def generate_synthetic_data():  # 二进制数据(?)
    """ Synthetic binary data """
    rs = np.random.RandomState(0)  # 产生相同的随机数
    n_pts = 36
    x, y = np.ogrid[0:l, 0:l]  # 快速产生一对数组, 一个横一个纵
    mask_outer = (x - l / 2.) ** 2 + (y - l / 2.) ** 2 < (l / 2.) ** 2
    mask = np.zeros((l, l))
    points = l * rs.rand(2, n_pts)
    mask[(points[0]).astype(np.int), (points[1]).astype(np.int)] = 1
    mask = ndimage.gaussian_filter(mask, sigma=l / n_pts)
    res = np.logical_and(mask > mask.mean(), mask_outer)
    return np.logical_xor(res, ndimage.binary_erosion(res))

#####################################################
# 生成合成图像, 并投影
l = 128
proj_operator = build_projection_operator(l, l // 5)
data = generate_synthetic_data()
proj = proj_operator * data.ravel()[:, np.newaxis]
proj += 0.15 * np.random.randn(*proj.shape)

# 使用L2正则项
ridge = linear_model.Ridge(alpha = 0.2)
ridge.fit(proj_operator, proj.ravel())
rec_l2 = ridge.coef_.reshape(l,l)

# 使用L1正则项, 留一法交叉验证 找到的最优alpha值
lasso = linear_model.Lasso(alpha=0.001)
lasso.fit(proj_operator, proj.ravel())
rec_l1 = lasso.coef_.reshape(l, l)

# 作图
plt.figure(figsize=(8, 3.3))
plt.subplot(131)
plt.imshow(data, cmap=plt.cm.gray, interpolation='nearest')
plt.axis('off')
plt.title('original image')
plt.subplot(132)
plt.imshow(rec_l2, cmap=plt.cm.gray, interpolation='nearest')
plt.title('L2 penalization')
plt.axis('off')
plt.subplot(133)
plt.imshow(rec_l1, cmap=plt.cm.gray, interpolation='nearest')
plt.title('L1 penalization')
plt.axis('off')

plt.subplots_adjust(hspace=0.01, wspace=0.01, top=1, bottom=0, left=0,
                    right=1)

plt.show()
