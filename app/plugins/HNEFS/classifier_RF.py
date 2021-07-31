#!/user/bin/env/ python
# -*- coding:utf-8 -*-

# @Time   : 2021/4/7 18:33
# @Author : Dong
# @File   : classifier_RF.py

import numpy as np
from time import time
from operator import truediv
import scipy.io as sio
import os
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsOneClassifier
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import accuracy_score



# 数据读取(这里是绝对路径)
salinas_path = r'G:\projects\doing\42667pyqt900 4-17\file_for_trans\salinas_corrected.mat'
salinas_gt_path = r'G:\projects\doing\42667pyqt900 4-17\file_for_trans\salinas_gt.mat'

indian_path = r'G:\projects\doing\42667pyqt900 4-17\file_for_trans\indian_pines_corrected.mat'
indian_gt_path = r'G:\projects\doing\42667pyqt900 4-17\file_for_trans\indian_pines_gt.mat'


# readimageInstance = ReadImage()
# img_array = readimageInstance.funReadMat(imagePath=indian_path) #型为（xxx,xxx,features）
# gt_array = readimageInstance.funReadMat(imagePath=indian_gt_path)
# # reshape为可直接输入的数据集，矩阵型为（样本个数，特征数个）
# X = img_array.reshape((-1, img_array.shape[-1]))
# # 类标
# y = gt_array.reshape((-1, ))


def classifierRF(X, y, data_subset:list,  n_estimators=100, min_samples_split=2, min_samples_leaf=1,  test_size=0.33):
	if data_subset!=list():
		X = X[:, data_subset]

	sfs = RandomForestClassifier(n_estimators=n_estimators, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf)

	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, shuffle=True, random_state=0)

	clf = OneVsOneClassifier(sfs).fit(X_train, y_train)

	y_pre = clf.predict(X_test)

	cf = confusion_matrix(y_true=y_test, y_pred=y_pre, labels=[x for x in range(0, 17)])

	list_diag = np.diag(cf)

	list_raw_sum = np.sum(cf, axis=1)

	each_acc = np.nan_to_num(truediv(list_diag, list_raw_sum))
	# 返回每个类的分类acc，类型array

	oa = accuracy_score(y_true=y_test, y_pred=y_pre)

	kc = cohen_kappa_score(y1=y_test, y2=y_pre, labels=[x for x in range(0, 17)])
	return each_acc, oa, kc

# 分类器小demo，这里的data_subset是一个特征的index，格式list，相当于将原始数据集中这几列提取出来，进行分类任务
# 需求里提到可以用没有data_subset的情况，就是直接用原始数据集进行分类，也就是包含全部特征，我在函数里第一个判断就是做这个事情
# 所以需要想一下这个地方界面怎么做比较好，就是可以提供输入几个数字作为list，也可以不输入直接拿读取的数据集分类
if __name__ == '__main__':

	a = np.random.randint(0, 4, (4, 10))
	print(a)