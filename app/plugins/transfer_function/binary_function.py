#!/user/bin/env/ python
# -*- coding:utf-8 -*-

# @Time   : 2021/5/18 15:27
# @Author : Dong
# @File   : binary_function.py

import numpy as np
import random
import math


def binary_function(transfered_array, type='v', **kwargs):
	# print(kwargs['pop'], kwargs['dim'])
	transfered_array = transfered_array.reshape((kwargs['pop'], kwargs['dim']))
	array_size = transfered_array.shape
	rand_array = np.random.uniform(0, 1, array_size)
	result_array = np.zeros(array_size)
	if type == 'v':
		# food position
		result_array = np.where(rand_array <= transfered_array, 1, 0)
		for index in range(array_size[0]):
			while np.sum(result_array[index]) == 0:
				result_array[index] = np.random.randint(0, 2, (1, kwargs['dim']))
	return result_array


if __name__ == '__main__':
	random_a = np.random.uniform(0, 1, (3, 5))
	result = binary_function(random_a)
	print('Input array:', random_a)
	print(result)
