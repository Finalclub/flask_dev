#!/user/bin/env/ python
# -*- coding:utf-8 -*-

# @Time   : 2021/5/17 12:48
# @Author : Dong
# @File   : trans_function.py

import math
import random
import numpy as np

def trans_func(x, type = 1):
	size = x.shape
	result = np.zeros(size)
	if type == 1:
		# sigmoid 1
		result = 1 / (1+np.exp(-10*(x-0.5)))
	elif type == 2:
		# sigmoid 2
		result = 1 / (1+np.exp(-x))
	elif type == 3:
		# va1
		result = np.abs(np.tanh(x)) / math.tanh(4)
	elif type == 4:
		# va2
		result = np.abs(math.sqrt(17)*x/(4*np.sqrt(1+np.power(x, 2))))
	elif type == 5:
		#va3
		result = np.abs(np.arctan(2/math.pi*x)/math.atan(2*math.pi))
	elif type == 6:
		#v1
		result = np.abs(np.tanh(x))
	elif type == 7:
		#v2
		result = np.abs(x / np.sqrt(1+np.power(x, 2)))
	elif type == 8:
		#v3
		coefficience = 2/math.pi
		result = np.abs(coefficience*np.arctan(coefficience*x))

	return result


if __name__ == '__main__':
	array1 = np.random.randint(0, 5, (1, 10))
	# print('array1: ', 5)
	# a = trans_func(array1, type=1)
	# print('Trans:', a)
	print(array1 * 0.5)
	# print(np.exp(array1))
	# print(trans_func(array1))