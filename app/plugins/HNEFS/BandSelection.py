#!/user/bin/env/ python
# -*- coding:utf-8 -*-

# @Time   : 2021/4/21 15:06
# @Author : Dong
# @File   : BandSelection.py

from HNEFS import NEFS
from time import time
from tqdm import tqdm
import readimage as ri


def find_min_index(dict_input:dict):
	min_key = min(dict_input, key=dict_input.get)
	return min_key


def bandSelection(X, Y, k, w, L, n_bands, return_reduction=False):
	'''
	:param X: reshaped source data
	:param Y: reshaped ground truth
	:param n_bands: bands need to be selected
	:param return_reduction: if returned dataset reduction
	:return: reduction of bandset index
	'''
	# 	初始化结果集
	S = list()
	N = len(X)
	D = X.shape[-1]

	band_index_list = [x for x in range(D)]

	for i in range(n_bands):
	# while len(S) < n_bands:
		set_for_search = set(band_index_list) - set(S)
		ne_rank_dict = dict()
		tic = time()
		# print('set_for_search', set_for_search)
		for band in tqdm(set_for_search):
			# print('band', band)
			nefs_band = S + [band]
			# print('nefs_band', nefs_band)
			# 计算并记录NE
			ne_rank_dict[str(nefs_band)] = NEFS(inputarray=X[:, nefs_band], inputgt=Y, k=k, w=w, L=L)
		# 获取最小NE值的key
		key_min = find_min_index(ne_rank_dict)
		# print('key_min', key_min)
		# key_min = min(ne_rank_dict.keys(), key=(lambda k: nefs_band[k]))
		key_min = key_min.replace('[', '').replace(']', '')
		# 更新S
		S = list(set(key_min.split(',') + S))
		S = [int(x) for x in S]
		S = list(set(S))
		toc = time()
		# print('lens{}, nbands{}'.format(len(S), n_bands))
		print('Already selected :{} ---> {}s'.format(str(S).replace('[', '').replace(']', ''), toc - tic))
	if return_reduction:
		return S, X[:, S]
	return S


if __name__=='__main__':
	salinas_path = r'D:\Projections\datasource\hyperimage\Salinas\salinas_corrected.mat'
	salinas_gt_path = r'D:\Projections\datasource\hyperimage\Salinas\salinas_gt.mat'

	indian_path = r'D:\Projections\datasource\hyperimage\Indian_pines\indian_pines_corrected.mat'
	indian_gt_path = r'D:\Projections\datasource\hyperimage\Indian_pines\indian_pines_gt.mat'

	iris = r'D:\Projections\datasource\commons\iris.csv'
	iris_lb = r'D:\Projections\datasource\commons\iris_class.csv'
	# img_array = sio.loadmat(salinas_path)

	# print(img_array)
	readimageInstance = ri.ReadImage()

	img_array = readimageInstance.funReadMat(imagePath=iris)
	gt_array = readimageInstance.funReadMat(imagePath=iris_lb)

	# reshape
	# img_array_reshaped = img_array.reshape((-1, img_array.shape[-1]))
	gt_array_reshaped = gt_array.reshape((-1,))

	print(bandSelection(X=img_array, Y=gt_array_reshaped, k=20, w=5, L=30, n_bands=2, return_reduction=False))