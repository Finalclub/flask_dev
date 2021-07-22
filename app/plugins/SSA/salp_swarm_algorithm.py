#!/user/bin/env/ python
# -*- coding:utf-8 -*-

# @Time   : 2021/5/19 13:50
# @Author : Dong
# @File   : salp_swarm_algorithm.py

import os
import numpy as np
np.set_printoptions(threshold=np.inf)
from tqdm import tqdm
from operator import truediv
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import accuracy_score
from position_init import position_inti as ps
from binary_function import binary_function as bf
from trans_function import trans_func
from position_update import food_position_update, position_update
from HNEFS import NEFS
from readimage import ReadImage as ri


def fitness_fucntion(X, y, test_size, all_size, if_details=False, return_all=False):
	# init
	# select dims
	length = X.shape[1]
	neigh = KNeighborsClassifier()
	# fitness_value
	fitness = None
	x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=test_size, shuffle=True, random_state=0)
	neigh.fit(X=x_train, y=y_train)
	y_pre = neigh.predict(X=x_test)
	if if_details and return_all:
		labels = set(y_train) & set(y_test)
		# confusion_matrix
		cf = confusion_matrix(y_true=y_test, y_pred=y_pre, labels=[x for x in labels])
		list_diag = np.diag(cf)
		list_raw_sum = np.sum(cf, axis=1)

		each_acc = np.nan_to_num(truediv(list_diag, list_raw_sum))

		oa = accuracy_score(y_true=y_test, y_pred=y_pre)

		kc = cohen_kappa_score(y1=y_test, y2=y_pre, labels=labels)

	accuracy = neigh.score(X=x_test, y=y_test)
	error = 1 - accuracy
	NE = NEFS(inputarray=X, inputgt=y)

	fitness = error + length / all_size + NE

	if return_all:
		return fitness, each_acc, oa, kc
	return fitness


def fitness_calcuate(binary_population, dataset, labels, **kwargs):
	fitness_dict = dict()
	row_number = 0
	for row in binary_population:
	# 	获取选取波段的索引
		band_index = [x for x, y in list(enumerate(row)) if y == 1]
		input_data = dataset[:, band_index]
		# 计算fitness value
		fitness = fitness_fucntion(X=input_data, y=labels, test_size=kwargs['test_size'], all_size=kwargs['all_size'])
		fitness_dict[row_number] = fitness
		row_number += 1
	return fitness_dict


def BSSA(n_population, max_iter, dataset, labels, trans_type, ub, lb):
	# defination
	dim = dataset.shape[1]
	# init
	cur_iter = 0
	salps_array = ps(agents_number=n_population, dim=dim, ub=ub, lb=lb, type='salp')
	# binary
	binary_array = bf(transfered_array=salps_array, type='v')
	# 计算第一次fitness，获取food position
	fitness_dict = fitness_calcuate(binary_population=binary_array, dataset=dataset, labels=labels, test_size=0.9, all_size=dim)
	# food position
	food_index = min(fitness_dict, key=fitness_dict.get)
	food_position = binary_array[food_index]
	best_fitness = fitness_dict[food_index]
	# follwers position
	followers_set = set([x for x in fitness_dict.keys()])
	followers_index = [x for x in followers_set if x != food_index]
	followers_position = binary_array[followers_index, :]
	# first time updateed position
	updated_food_position = food_position_update(position_array=food_position, ub=ub, lb=lb, type='salp', t=cur_iter,
												 max_t=max_iter)
	updated_food_position = updated_food_position.reshape((1, -1))
	updated_salps_array = np.concatenate((updated_food_position, followers_position), axis=0)
	updated_positions = position_update(positions=updated_salps_array, type='salp')

	while cur_iter < max_iter:
		print(cur_iter)
		# transfer function
		trans_position = trans_func(updated_positions, type=trans_type)
		# binary position
		updated_binary_position = bf(transfered_array=trans_position)
		# updated_binary_food_position = updated_binary_food_position.reshape((1, -1))
		# updated_binary_follower_position = bf(transfered_array=trans_follower_position)
		#
		# input_binary_array = np.concatenate((updated_binary_food_position, updated_binary_follower_position), axis=0)
		# print(input_binary_array)
		fitness_dict = fitness_calcuate(binary_population=updated_binary_position, dataset=dataset, labels=labels, test_size=0.2, all_size=dim)
		temp_best_index = min(fitness_dict, key=fitness_dict.get)
		temp_best_fitness = fitness_dict[temp_best_index]

		# follwers position
		followers_set = set([x for x in fitness_dict.keys()])
		followers_index = [x for x in followers_set if x != temp_best_index]
		followers_position = binary_array[followers_index, :]
		if temp_best_fitness < best_fitness:
			food_index = temp_best_index
			food_position = updated_binary_position[food_index]
			best_fitness = temp_best_fitness
		cur_iter += 1
		print('Current best:', food_position)
		print('Current best fitness:', best_fitness)
		# updated food position
		lead_position = updated_binary_position[temp_best_index]
		updated_food_position = food_position_update(position_array=lead_position, ub=ub, lb=lb, type='salp', t=cur_iter, max_t=max_iter)
		updated_food_position = updated_food_position.reshape((1, -1))
		updated_salps_array = np.concatenate((updated_food_position, followers_position), axis=0)
		updated_positions = position_update(positions=updated_salps_array, type='salp')

	selected_feature_index = [x for x, y in list(enumerate(food_position)) if y == 1]

	return selected_feature_index, best_fitness


def expriment_function(dataset, labels, population, max_iter, times, ub, lb, trans_type, file_path):

	dict_total_result = dict()
	cur_time = 1
	for _ in tqdm(range(times)):
		print('Current time:', cur_time)
		print('Now is {}'.format(os.path.basename(file_path)))
		selected, best_fitness = BSSA(n_population=population, max_iter=max_iter, dataset=dataset, labels=labels, trans_type=trans_type, ub=ub, lb=lb)
		dict_total_result[best_fitness] = selected
		with open(r'{}'.format(file_path), 'a+') as f:
			f.write(str(dict_total_result))
		cur_time += 1
	return dict_total_result


if __name__ == '__main__':
	salinas_path = r'D:\Projections\datasource\hyperimage\Salinas\salinas_corrected.mat'
	salinas_gt_path = r'D:\Projections\datasource\hyperimage\Salinas\salinas_gt.mat'

	indian_path = r'D:\Projections\datasource\hyperimage\Indian_pines\indian_pines_corrected.mat'
	indian_gt_path = r'D:\Projections\datasource\hyperimage\Indian_pines\indian_pines_gt.mat'

	# img_array = sio.loadmat(salinas_path)

	# print(img_array)
	readimageInstance = ri()

	img_array = readimageInstance.funReadMat(imagePath=indian_path)
	gt_array = readimageInstance.funReadMat(imagePath=indian_gt_path)

	# reshape
	img_array_reshaped = img_array.reshape((-1, img_array.shape[-1]))
	gt_array_reshaped = gt_array.reshape((-1,))

	save_path = r'D:\Papers\result_collection\heuristic_algorithm\SSA\{}'.format('ip_trans_v2_2.txt')
	reseult_dict = expriment_function(dataset=img_array_reshaped, labels=gt_array_reshaped, population=50, max_iter=50, times=11, trans_type=4, ub=1, lb=0, file_path=save_path)
	print(reseult_dict)
