#!/user/bin/env/ python
# -*- coding:utf-8 -*-

# @Time   : 2021/5/18 15:58
# @Author : Dong
# @File   : position_update.py

import numpy as np
import math
from app.plugins.transfer_function.binary_function import binary_function as bf


def chaotic_map(array_size, type='logistic'):
	updated_array = None
	if type == 'logistic':
		c = 4
		updated_array = c * array_size * (1 - array_size)
	return updated_array


def food_position_update(position_array, ub, lb, type='salp', **kwargs):
	array_size = position_array.shape
	result_updated_position = position_array
	if type == 'salp':
		# food position
		c1 = 2 * math.exp(-math.pow(4*kwargs['t']/kwargs['max_t'], 2))
		c2_random_array = np.random.uniform(0, 1, array_size)
		c3_random_array = np.random.uniform(0, 1, array_size)
		# print('C3:', c3_random_array)
		result_updated_position = np.where(c3_random_array >= 0.5, position_array + c1*((ub - lb)*c2_random_array+lb), result_updated_position)
		# print('Result ba 1:', result_updated_position)
		result_updated_position = np.where(c3_random_array < 0.5, position_array - c1*((ub - lb)*c2_random_array+lb), result_updated_position)
	elif type == 'quantum_salp':
		# food position
		c1 = 2 * math.exp(-math.pow(4 * kwargs['t'] / kwargs['max_t'], 2))
		c2_updated_array = np.random.uniform(0, 1, array_size)
		c3_random_array = np.random.uniform(0, 1, array_size)
		kwargs['all_position_array'] = kwargs['all_position_array'].reshape((-1, max(kwargs['all_position_array'].shape)))
		M = np.sum(kwargs['all_position_array']) / kwargs['population']
		result_updated_position = np.where(c3_random_array >= 0.5, position_array + c1*(M - position_array)*c2_updated_array, result_updated_position)
		# print('Result ba 1:', result_updated_position)
		result_updated_position = np.where(c3_random_array < 0.5, position_array - c1*(M - position_array)*c2_updated_array, result_updated_position)
	elif type == 'quantum&chaos':
		# food position
		c1 = 2 * math.exp(-math.pow(4 * kwargs['t'] / kwargs['max_t'], 2))
		c2_random_array = kwargs['c2']
		c3_random_array = np.random.uniform(0, 1, array_size)
		kwargs['all_position_array'] = kwargs['all_position_array'].reshape((-1, max(kwargs['all_position_array'].shape)))
		M = np.sum(kwargs['all_position_array']) / kwargs['population']
		result_updated_position = np.where(c3_random_array >= 0.5, position_array + c1*(M - position_array)*c2_random_array, result_updated_position)
		# print('Result ba 1:', result_updated_position)
		result_updated_position = np.where(c3_random_array < 0.5, position_array - c1*(M - position_array)*c2_random_array, result_updated_position)

	return result_updated_position


def position_update(positions, type='salp', **kwargs):
	array_size = positions.shape
	updated_array = positions
	first_follower = positions[0]
	if type == 'salp':

		for index in range(1, array_size[0]):
			updated_array[index] = (updated_array[index] + updated_array[index - 1]) / 2
			while np.sum(updated_array[index-1]) == 0:
				updated_array[index - 1] = np.random.randint(0, 2, (1, array_size[1]))
		updated_array[-1] = (first_follower + updated_array[-1]) / 2

	return updated_array


if __name__=='__main__':
	food = np.random.uniform(0, 1, (1, 5))
	print('food:', food)
	input_array = np.random.uniform(0, 1, (1, 5))
	print('followers:', input_array)
	# all_position = np.concatenate((food, input_array), axis=0)
	# print('Concate:', all_position)
	# updated_array = food_position_update(position_array=input_array, type='quantum&chaos', ub=1, lb=0, t=2, max_t=100, all_position_array=all_position, population=4, c2 = c2)
	# print(updated_array)
	print(food * input_array)
	# print(bf(updated_array, type='v'))