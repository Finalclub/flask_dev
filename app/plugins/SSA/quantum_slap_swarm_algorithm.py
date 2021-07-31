#!/user/bin/env/ python
# -*- coding:utf-8 -*-

# @Time   : 2021/5/25 18:25
# @Author : Dong
# @File   : quantum_slap_swarm_algorithm.py

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
from app.plugins.position_process.position_init import position_inti as ps
from app.plugins.transfer_function.binary_function import binary_function as bf
from app.plugins.transfer_function.trans_function import trans_func
from app.plugins.position_process.position_update import food_position_update, position_update, chaotic_map
from app.plugins.HNEFS.HNEFS import NEFS
from app.plugins.LSH.readimage import ReadImage


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
	NE = NEFS(inputarray=X, inputgt=y, k=20, w=5, L=30)
	# print('error:', error)
	# print('length:{}, all_size:{}, ratio:{}'.format(length, all_size, length / all_size))
	# print('NE:', NE)

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


def QSSA(n_population, max_iter, dataset, labels, trans_type, ub, lb):
	# defination
	dim = dataset.shape[1]
	# init
	cur_iter = 0
	updated_food_position = None
	trans_follower_position = None
	salps_array = ps(agents_number=n_population, dim=dim, ub=ub, lb=lb, type='salp')
	c2 = np.random.uniform(0, 1, (1, dim))
	# binary
	binary_array = bf(transfered_array=salps_array, type='v', dim=dim, pop=n_population)
	# 计算第一次fitness，获取food position
	fitness_dict = fitness_calcuate(binary_population=binary_array, dataset=dataset, labels=labels, test_size=0.2,
									all_size=dim)
	# food position
	food_index = min(fitness_dict, key=fitness_dict.get)
	food_position = binary_array[food_index].reshape((-1, max(binary_array[food_index].shape)))
	best_fitness = fitness_dict[food_index]
	# follwers position
	followers_set = set([x for x in fitness_dict.keys()])
	followers_index = [x for x in followers_set if x != food_index]
	followers_position = binary_array[followers_index, :]
	all_position = np.concatenate((food_position, followers_position), axis=0)

	# first time updateed position
	updated_food_position = food_position_update(position_array=food_position, type='quantum&chaos', ub=1, lb=0, t=cur_iter, max_t=max_iter, all_position_array=all_position, population=4, c2=c2)
	updated_food_position = updated_food_position.reshape((1, -1))
	updated_salps_array = np.concatenate((updated_food_position, followers_position), axis=0)
	updated_positions = position_update(positions=updated_salps_array, type='salp')

	while cur_iter < max_iter:
		print(cur_iter)
		c2 = chaotic_map(array_size=c2)
		# if cur_iter != 0:
		# 	all_position = np.concatenate((updated_binary_food_position, updated_binary_follower_position), axis=0)
		# transfer function
		trans_position = trans_func(updated_positions, type=trans_type)
		# binary position
		updated_binary_position = bf(transfered_array=trans_position, dim=dim, pop=n_population)
		# updated_binary_food_position = bf(transfered_array=trans_position)
		# updated_binary_food_position = updated_binary_food_position.reshape((1, -1))
		# updated_binary_follower_position = bf(transfered_array=trans_follower_position)

		# input_binary_array = np.concatenate((updated_binary_food_position, updated_binary_follower_position), axis=0)
		# print(input_binary_array)
		fitness_dict = fitness_calcuate(binary_population=updated_binary_position, dataset=dataset, labels=labels, test_size=0.2, all_size=dim)
		temp_best_index = min(fitness_dict, key=fitness_dict.get)
		temp_best_fitness = fitness_dict[temp_best_index]
		if temp_best_fitness < best_fitness:
			food_position = updated_binary_position[temp_best_index]
			best_fitness = temp_best_fitness
		cur_iter += 1
		# print('Current best:', food_position)
		# print('Current best fitness:', best_fitness)
		# updated positions
		lead_position = updated_binary_position[temp_best_index]
		followers_set = set([x for x in fitness_dict.keys()])
		followers_index = [x for x in followers_set if x != temp_best_index]
		followers_position = binary_array[followers_index, :]

		updated_food_position = food_position_update(position_array=lead_position, type='quantum&chaos', ub=1, lb=0, t=cur_iter, max_t=max_iter, all_position_array=all_position, population=n_population, c2=c2)
		updated_food_position = updated_food_position.reshape((1, -1))
		updated_salps_array = np.concatenate((updated_food_position, followers_position), axis=0)
		updated_positions = position_update(positions=updated_salps_array, type='salp')

	print((food_position))
	try:
		print(1)
		selected_feature_index = [x for x, y in enumerate(food_position[0].tolist()) if y == 1]
	except:
		print(2)
		selected_feature_index = [x for x, y in enumerate(food_position.tolist()) if y == 1]
	# print('selected_feature_index:', selected_feature_index)
	# print('best_fitness:', best_fitness)

	return selected_feature_index, best_fitness


def expriment_function(dataset, labels, population, max_iter, times, ub, lb, trans_type=7, file_path=os.path.dirname(__file__)):

	dict_total_result = dict()
	cur_time = 1
	for _ in tqdm(range(times)):
		print('Current time:', cur_time)
		print('Now is {}'.format(os.path.basename(file_path)))
		selected, best_fitness = QSSA(n_population=population, max_iter=max_iter, dataset=dataset, labels=labels, trans_type=trans_type, ub=ub, lb=lb)
		dict_total_result[best_fitness] = selected
		cur_time += 1
	with open(r'{}'.format(file_path), 'a+') as f:
		f.write(str(dict_total_result))
	return dict_total_result


if __name__ == '__main__':
	salinas_path = r'D:\Projections\datasource\hyperimage\Salinas\salinas_corrected.mat'
	salinas_gt_path = r'D:\Projections\datasource\hyperimage\Salinas\salinas_gt.mat'

	indian_path = r'D:\Projections\datasource\hyperimage\Indian_pines\indian_pines_corrected.mat'
	indian_gt_path = r'D:\Projections\datasource\hyperimage\Indian_pines\indian_pines_gt.mat'

	iris = r'D:\Projections\datasource\commons\iris.csv'
	iris_lb = r'D:\Projections\datasource\commons\iris_class.csv'

	# print(img_array)
	readimageInstance = ReadImage()

	img_array = readimageInstance.funReadMat(imagePath=iris)
	gt_array = readimageInstance.funReadMat(imagePath=iris_lb)

	# reshape
	# img_array_reshaped = img_array.reshape((-1, img_array.shape[-1]))
	gt_array_reshaped = gt_array.reshape((-1,))

	save_path = r'D:\Papers\result_collection\heuristic_algorithm\QSSA\{}'.format('50_100_ip_v3_for_test_2.txt')
	reseult_dict = expriment_function(dataset=img_array, labels=gt_array_reshaped, population=50, max_iter=50, trans_type=7, times=10, ub=1, lb=0, file_path=save_path)
	print(reseult_dict)
