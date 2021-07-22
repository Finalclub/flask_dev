#!/usr/bin/env python3
# _*_ coding:utf-8 _*_
# @Version : 1.0
# @Time    : 2019/4/4 22:15
# @Author  : Dong
# @File    : readimage.py

# from PIL import Image
import scipy.io as sio
import os
import numpy as np


class ReadImage(object):
	"""docstring for ReadImage"""
	def funReadMat(self, imagePath):
		format = os.path.splitext(imagePath)[-1]
		try:
			if format == '.mat':
				self.dataName = os.path.basename(imagePath).split('.')[0]
				# 读取matlib格式文件,目前已获取数据的结构为字典
				self.matArray = sio.loadmat(imagePath)
				#针对目前的数据集采用特定的方式提取矩阵
				self.hyperspectrumData = self.matArray[self.dataName]
				# reshape 为二维矩阵
				shape = self.hyperspectrumData.shape
				if shape[0] == shape[-1]:
					self.hyperspectrumData = self.hyperspectrumData.reshape((-1, 1))
				else:
					self.hyperspectrumData = self.hyperspectrumData.reshape((-1, shape[-1]))
				return self.hyperspectrumData
			elif format == '.csv':
				self.dataName = os.path.basename(imagePath).split('.')[0]
				self.csvArray = np.loadtxt(imagePath, delimiter=',')
				# print(self.csvArray)
				shape = self.csvArray.shape
				# print(shape)
				if shape[0] == shape[-1]:
					self.hyperspectrumData = self.csvArray.reshape((-1, 1))
				else:
					self.hyperspectrumData = self.csvArray.reshape((-1, shape[-1]))
				return self.hyperspectrumData
		except:
			print('Check out file format or path of file.')


if __name__ == '__main__':
	indian_path = r'D:\Projections\datasource\hyperimage\Indian_pines\indian_pines_corrected.mat'
	indian_gt_path = r'D:\Projections\datasource\hyperimage\Indian_pines\indian_pines_gt.mat'
	iris = r'D:\Projections\datasource\commons\iris_class.csv'
	ri = ReadImage()
	# array = ri.funReadMat(indian_path)
	# print(array)
	matarray = ri.funReadMat(iris)
	print(matarray)