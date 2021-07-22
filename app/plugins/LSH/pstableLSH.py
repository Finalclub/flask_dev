#!/user/bin/env/ python
# -*- coding:utf-8 -*-

# @Time   : 2021/4/20 17:30
# @Author : Dong
# @File   : pstableLSH.py

from sklearn.neighbors import NearestNeighbors


def pstableSLSH(k, w, L, x):
	'''

	:param k: k nn
	:param w: 量化系数
	:param L: L tables
	:param x: data source
	:return:
	'''
	neigh = NearestNeighbors(n_neighbors=k, radius=w, leaf_size=L, algorithm='kd_tree')
	neigh.fit(x)

	return neigh

def LSHquery(query, k_nn, index):

	sample_index = index.kneighbors(query, n_neighbors=k_nn, return_distance=False)

	return sample_index[0].tolist()


if __name__ == '__main__':
	pass