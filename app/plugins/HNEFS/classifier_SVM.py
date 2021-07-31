#!/user/bin/env/ python
# -*- coding:utf-8 -*-

# @Time   : 2021/4/7 18:30
# @Author : Dong
# @File   : classifier_SVM.py

import numpy as np
from operator import truediv
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsOneClassifier
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import accuracy_score
from app.plugins.LSH.readimage import ReadImage


def classifierSVM(X, y, data_subset: list,  kernel='rbf', test_size=0.33):

	if data_subset != list():
		data_subset = [int(x) for x in data_subset]
		X = X[:, data_subset]

	sfs = SVC(kernel=kernel, probability=True)

	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, shuffle=True, random_state=0)

	clf = OneVsOneClassifier(sfs).fit(X_train, y_train)

	y_pre = clf.predict(X_test)

	labels = list(set(y.tolist()))
	# print(labels)

	cf = confusion_matrix(y_true=y_test, y_pred=y_pre, labels=labels)

	list_diag = np.diag(cf)

	list_raw_sum = np.sum(cf, axis=1)

	each_acc = np.nan_to_num(truediv(list_diag, list_raw_sum))
	# 返回每个类的分类acc，类型array

	oa = accuracy_score(y_true=y_test, y_pred=y_pre)

	kc = cohen_kappa_score(y1=y_test, y2=y_pre, labels=labels)
	return each_acc.tolist(), oa, kc

'''分类器小demo，这里的data_subset是一个特征的index，格式list，相当于将原始数据集中这几列提取出来，进行分类任务
 需求里提到可以用没有data_subset的情况，就是直接用原始数据集进行分类，也就是包含全部特征，我在函数里第一个判断就是做这个事情
 所以需要想一下这个地方界面怎么做比较好，就是可以提供输入几个数字作为list，也可以不输入直接拿读取的数据集分类
 函数里还有两个关键字命名的参数，也在界面的需要呈现
 '''

if __name__ == '__main__':
	iris = r'D:\Projections\datasource\commons\iris.csv'
	iris_lb = r'D:\Projections\datasource\commons\iris_class.csv'

	indian_path = r'D:\Projections\datasource\hyperimage\Indian_pines\indian_pines_corrected.mat'
	indian_gt_path = r'D:\Projections\datasource\hyperimage\Indian_pines\indian_pines_gt.mat'

	readimageInstance = ReadImage()

	img_array = readimageInstance.funReadMat(imagePath=iris)
	gt_array = readimageInstance.funReadMat(imagePath=iris_lb)

	# reshape
	# img_array_reshaped = img_array.reshape((-1, img_array.shape[-1]))
	gt_array = gt_array.reshape((-1,))

	result_array, oa, kc = classifierSVM(img_array, gt_array, data_subset=[3])
	print(result_array, oa, kc)