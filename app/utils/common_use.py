#!/user/bin/env/ python
# -*- coding:utf-8 -*-

# @Time   : 2021/7/21 17:46
# @Author : Dong
# @File   : common_use.py

from app.utils.SQL import SQLHelper
from app.plugins.LSH.readimage import ReadImage
import json
import numpy as np


def get_algorithm_args(form, type:str):
	d_id = form.dataset.data
	l_id = form.labels.data
	dataset_path = SQLHelper.fetch_one('select file_path from dataset where id = {}'.format(d_id))
	label_path = SQLHelper.fetch_one('select file_path from dataset where id = {}'.format(l_id))
	if type == 'ne':
		return dataset_path, label_path, form.k.data, \
				form.w.data, form.L.data, form.n_bands.data
	elif type == 'ssa':
		return dataset_path, label_path, form.pop.data, \
				form.max_iter.data, form.times.data
	elif type == 'svm':
		return dataset_path, label_path, form.test_set.data, \
				form.subset.data
	elif type == 'rf':
		return dataset_path, label_path, form.test_set.data, \
				form.subset.data, form.n_estimators.data, \
				form.min_samples_split.data, form.min_samples_leaf.data
	elif type == 'knn':
		return dataset_path, label_path, form.test_set.data, \
				form.subset.data, form.n_neighbors.data


def get_data(path1, path2):
	ri = ReadImage()
	img_array = ri.funReadMat(imagePath=path1)
	gt_array = ri.funReadMat(imagePath=path2)

	gt_array = gt_array.reshape((-1, ))
	return img_array, gt_array


def find_min_index(dict_input:dict):
	min_key = min(dict_input, key=dict_input.get)
	print(min_key)
	return min_key


def to_json(array):
	dict_turn = dict()
	dict_turn['index'] = array.tolist()
	return json.dumps(dict_turn)


def to_array(json_input):
	array = np.array(json.loads(json_input)['index'])
	return array