#!/user/bin/env/ python
# -*- coding:utf-8 -*-

# @Time   : 2021/7/23 22:53
# @Author : Dong
# @File   : classifier_KNN.py

import numpy as np
from operator import truediv
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsOneClassifier
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import accuracy_score


def classifierRF(X, y, data_subset:list,  test_size=0.2, k=5):
	if data_subset!=list():
		X = X[:, data_subset]

	sfs = KNeighborsClassifier(n_neighbors=k)

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