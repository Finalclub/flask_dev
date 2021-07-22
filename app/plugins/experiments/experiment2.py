#!/usr/bin/env python3
# _*_ coding:utf-8 _*_
# @Version : 1.0
# @Time    : 2020/3/10 9:00
# @Author  : zhaidongchang
# @File    : experiment1.py

import distanceMetricLSH as dmlsh
import readimage as ri
import pandas as pd
import numpy as np
import math
import datetime
from RewriteThread import RewriteThread as RT

datapath = {
	'indian': 'D:/Projections/datasource/hyperimage/Indian_pines/Indian_pines_corrected.csv',
	'Salinas': 'D:/Projections/datasource/hyperimage/Salinas/salinas_corrected.mat'
	}

gtpath = {
	'indian': 'D:/Projections/datasource/hyperimage/Indian_pines/Indian_pines_gt.csv',
	'Salinas': 'D:/Projections/datasource/hyperimage/Salinas/salinas_gt.mat'
	}

# readInstance = ri.ReadImage()

spectralImage = pd.read_csv(datapath['indian'])
spectralImage = spectralImage.values
groundTruth = pd.read_csv(gtpath['indian'])
groundTruth = groundTruth.values
# get label statistic dict
labeldict = {}
for pixel in groundTruth:
	labeldict[pixel[0]] = labeldict.get(0, pixel[0]) + 1


def funUpdateArray(dataframe, indexpop):
	# 输入为加入标签的特征子集array
	# 已经进入样本邻域子集的列表indexpop
	# 返回去邻域子集索引的array，即将邻域子集中的样本从原样本集合中去除
	updatedarray = dataframe[~dataframe.index.isin(indexpop)]
	return updatedarray.values


def funRelativeFreq(subset, label):
	# 计算相对频率
	# 输入参数为邻域子集subset，输入的类标label
	# 返回的是当前类标和邻域子集下的相对频率
	neighbors = len(subset)
	if neighbors != 0:
		n = 0
		for nn in subset:
			if groundTruth[nn][0] == label:
				n += 1
		return n / neighbors
	else:
		return 0


def funNE(array):
	# 输入为特征子集，返回当前特征子集计算得到的熵
	# 构建lsh索引
	LSH = dmlsh.DMLSH(10, 4, 10)
	LSH.funDistanceMetricLSHIndex(array, inputPattern='Pixel')
	# 记录样本个数
	numofsamples = len(array)
	# 该特征子集熵的值初始化为0
	entropy = 0
	# 初始化记录弹出样本索引的list以及用于更新array的dataframe
	poplist = []
	origindataframe = pd.DataFrame(array)
	while len(array) != 0:
		# 进行最近邻搜索
		nns = LSH.funNNSearch(array[0])
		for index in nns:
			poplist.append(index)
		array = funUpdateArray(origindataframe, poplist)
		# labeldict为全局变量
		for label in labeldict.keys():
			relativeFreq = funRelativeFreq(nns, label)
			if relativeFreq != 0:
				# 计算单个类标下相对频率计算的结果，并进行累加
				entropy += (relativeFreq * math.log(relativeFreq))
	# 返回输入特征的NE
	return - entropy / numofsamples


def funIntegratedExperiment(dataset, numofbands):
	# 构建整合实验，输入为整个数据集dataset
	# 输入为要选择的波段个数
	# 记录并添加每个波段的序号
	numoffeatures = dataset.shape[-1]
	# bandsequence = [num for num in range(numoffeatures)]
	# 将数据集dataframe化
	dataset = pd.DataFrame(dataset)
	# dataset = dataset.append([bandsequence], ignore_index=True)
	# 初始化波段选择子集
	bandset = []
	while len(bandset) < numofbands:
		# 初始化最小NE记录的列表，index=0存放NE值，index=1存放相应的波段序号
		minNE = [0, 0]
		if len(bandset) == 0:
			for i in range(numoffeatures):
				inputarray = dataset.iloc[:, [i]].values
				entropy = funNE(inputarray)
				if minNE[0] == 0:
					minNE[0] = entropy
					minNE[1] = i
				elif entropy < minNE[0]:
					minNE[0] = entropy
					minNE[1] = i
		else:
			for i in range(numoffeatures):
				if i not in bandset:
					bandset.append(i)
					inputarray = dataset.iloc[:, bandset].values
					bandset.pop(-1)
					entropy = funNE(inputarray)
					if minNE[0] == 0:
						minNE[0] = entropy
						minNE[1] = i
					elif entropy < minNE[0]:
						minNE[0] = entropy
						minNE[1] = i
		bandset.append(minNE[1])
		print('Selected {} bands in {}'.format(len(bandset), datetime.datetime.now()))
		# 提取选取的波段
	return bandset


if __name__ == '__main__':
	threadList = []
	resultCollector = []
	for bands in range(5, 46, 5):
		subThread = RT(funIntegratedExperiment, args=(spectralImage, bands))
		threadList.append(subThread)
		subThread.start()
	for thread in threadList:
		thread.join()
		resultCollector.append(thread.funResult())
	with open('', 'r+') as file:
		file.writelines(str(resultCollector))