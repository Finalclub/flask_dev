#!/usr/bin/env python3
# _*_ coding:utf-8 _*_
# @Version : 1.0
# @Time    : 2019/4/22 9:00
# @Author  : zhaidongchang
# @File    : indexing.py

import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
# from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import KFold
from sklearn.naive_bayes import GaussianNB
# from sklearn.metrics import roc_auc_score
from RewriteThread import RewriteThread as rt

datapath = {
	'indian': 'D:/Projections/datasource/hyperimage/indian_pines/indian_pines_corrected.csv',
	'Salinas': 'D:/Projections/datasource/hyperimage/indian_pines/indian_pines_corrected.mat'
	}

gtpath = {
	'indian': 'D:/Projections/datasource/hyperimage/indian_pines/indian_pines_gt.csv',
	'Salinas': 'D:/Projections/datasource/hyperimage/indian_pines/indian_pines_gt.mat'
	}

# spectralImage = pd.read_csv(datapath['indian'])
# spectralImage = spectralImage.astype(np.int64)
# reshapedImage = spectralImage.values.reshape(-1, spectralImage.shape[-1])
# reshapedImage = pd.DataFrame(reshapedImage)
# groundTruth = pd.read_csv(gtpath['indian'])
print(type(3 / 4))