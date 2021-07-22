#!/user/bin/env/ python
# -*- coding:utf-8 -*-

# @Time   : 2021/4/20 17:06
# @Author : Dong
# @File   : HNEFS.py


import numpy as np
from time import time
import math
from tqdm import tqdm
from app.plugins.LSH.readimage import ReadImage
from app.plugins.LSH.pstableLSH import pstableSLSH as plsh, LSHquery


def recall(true_label, result_list):
	counter = 0
	for item in result_list:
		if item == true_label:
			counter += 1
	return counter / len(result_list)


def relativeFreq(nnlist, inputgt):
	'''

	:param nnlist:
	:param c: true label
	:return:
	'''
	nn = len(nnlist)
	dict_1 = dict()
	label_list = inputgt[nnlist]
	item_set = set(label_list)
	result = 0
	for label in label_list:
		dict_1[label] = dict_1.get(label, 0) + 1
	for label in item_set: #计算样本邻域的相关频率
		fr = dict_1[label]/nn
		result += fr*math.log(fr)
	return result


def NEFS(inputarray, inputgt, k, w, L, return_recall=False):
	'''

	:param inputarray:
	:param inputgt:
	:return:
	'''
	N = len(inputarray)

	avg_recall = 0

	# lsh = plsh.pstableSLSH(k=20, w=5, L=30, x=inputarray)
	lsh = plsh(k=k, w=w, L=L, x=inputarray)

	NE = 0
	for i in range(N):
		query = inputarray[i].reshape((1, -1))
		ture_label = inputgt[[i]]
		result_index_list = LSHquery(query=query, k_nn=20, index=lsh) #计算邻域集合list
		avg_recall += recall(ture_label, inputgt[result_index_list]) #计算累计recall
		fr_sub_item = relativeFreq(nnlist=result_index_list, inputgt=inputgt)
		NE += fr_sub_item
	if return_recall:
		return -NE/N, recall
	return -NE/N


if __name__=='__main__':
	# (2.99573227355536, 0.43028775267539116)
	salinas_path = r'D:\Projections\datasource\hyperimage\Salinas\salinas_corrected.mat'
	salinas_gt_path = r'D:\Projections\datasource\hyperimage\Salinas\salinas_gt.mat'

	indian_path = r'D:\Projections\datasource\hyperimage\Indian_pines\indian_pines_corrected.mat'
	indian_gt_path = r'D:\Projections\datasource\hyperimage\Indian_pines\indian_pines_gt.mat'

	# img_array = sio.loadmat(salinas_path)

	# print(img_array)
	readimageInstance = ReadImage()

	img_array = readimageInstance.funReadMat(imagePath=indian_path)
	gt_array = readimageInstance.funReadMat(imagePath=indian_gt_path)

	# reshape
	img_array_reshaped = img_array.reshape((-1, img_array.shape[-1]))
	gt_array_reshaped = gt_array.reshape((-1,))

	N = len(img_array_reshaped)
	D = img_array_reshaped.shape[-1]
	# result_list = list()
	# tic = time()
	# lsh = plsh.pstableSLSH(k=100, w=5, L=30, x=img_array_reshaped[:, [11]])

	# query = img_array_reshaped[:, [11, 25]][2].reshape((1, -1))
	# ture_label = gt_array_reshaped[2]
	# result_index_list = lsh.radius_neighbors(X=query, radius=20, return_distance=False)

	# rfr = relativeFreq(result_index_list[0].tolist(), inputgt=gt_array_reshaped)
	# print(rfr)
	# print(NEFS(img_array_reshaped[:, [11]], gt_array_reshaped))
	test_selected_bands_set = [15, 45, 66, 107]
	test_array = img_array_reshaped[:, test_selected_bands_set]
	print(test_array.shape)
	ne = NEFS(inputarray=test_array, inputgt=gt_array_reshaped)
	print(ne)

	# for i in range(D):
	# 	temp_val = NEFS(img_array_reshaped[:, [i]], gt_array_reshaped)
	# 	result_list.append(temp_val)
	# toc = time()
	# print(result_list)
	# print('{}s'. format(toc-tic))