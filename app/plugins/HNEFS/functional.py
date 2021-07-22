#!/user/bin/env/ python
# -*- coding:utf-8 -*-

# @Time   : 2021/4/20 17:18
# @Author : Dong
# @File   : functional.py


import math
import numpy as np
from time import time
import readimage as ri
from LSH import pstableLSH as plsh
from sklearn.neighbors import NearestNeighbors

def pstableSLSH(k, w, L, x):
	'''

	:param k: k nn
	:param w: 量化系数
	:param L: L tables
	:param x: data source
	:return:
	'''
	neigh = NearestNeighbors(n_neighbors=k, radius=w, leaf_size=L, algorithm='brute')
	neigh.fit(x)

	return neigh

def LSHquery(query, k_nn, index):

	sample_index = index.kneighbors(query, n_neighbors=k_nn, return_distance=False)

	return sample_index[0].tolist()


salinas_path = r'D:\Projections\datasource\hyperimage\Salinas\salinas_corrected.mat'
salinas_gt_path = r'D:\Projections\datasource\hyperimage\Salinas\salinas_gt.mat'

indian_path = r'D:\Projections\datasource\hyperimage\Indian_pines\indian_pines_corrected.mat'
indian_gt_path = r'D:\Projections\datasource\hyperimage\Indian_pines\indian_pines_gt.mat'

# img_array = sio.loadmat(salinas_path)

# print(img_array)
readimageInstance = ri.ReadImage()

img_array = readimageInstance.funReadMat(imagePath=indian_path)
gt_array = readimageInstance.funReadMat(imagePath=indian_gt_path)
# reshape
img_array_reshaped = img_array.reshape((-1, img_array.shape[-1]))
gt_array_reshaped = gt_array.reshape((-1, ))

# print(img_array_reshaped[:, [2]])

# print(img_array_reshaped[:, [0, 12, 15]])
# print('--------------')
# print(img_array_reshaped[:, [0, 12, 15]][2].reshape(1,-1))
# print('--------------')
# neigh = pstableSLSH(k=100, w=4, L=30, x=img_array_reshaped[:, [0]])
# result = LSHquery(query=img_array_reshaped[:, [0]][2].reshape(1, -1), k_nn=20, index=neigh)
# print(result)
# for i in range(len(img_array_reshaped)):
# 	print(LSHquery(query=img_array_reshaped[i, [0]))

a = [1,2,3,4,5,6]

b = 5
print([b])
print(type([b]))

array1 = np.random.uniform(0, 1, (1, 10))
print(array1)