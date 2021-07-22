#!/user/bin/env/ python
# -*- coding:utf-8 -*-

# @Time   : 2021/5/18 15:27
# @Author : Dong
# @File   : binary_function.py

import numpy as np
import math


def binary_function(transfered_array, type = 'v', **kwargs):
	transfered_array = transfered_array.reshape((-1, max(transfered_array.shape)))
	array_size = transfered_array.shape
	rand_array = np.random.uniform(0, 1, array_size)
	result_array = np.zeros(array_size)
	if type == 'v':
		# food position
		result_array = np.where(rand_array <= transfered_array, 1, 0)
		for index in range(array_size[0]):
			if np.sum(result_array[index]) == 0:
				result_array[index] = np.random.randint(0, 2, (1, array_size[1]))
	return result_array


if __name__ == '__main__':
	random_a = np.random.uniform(0, 1, (3, 5))
	result = binary_function(random_a)
	print('Input array:', random_a)
	print(result)
