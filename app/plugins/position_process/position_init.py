#!/user/bin/env/ python
# -*- coding:utf-8 -*-

# @Time   : 2021/5/18 14:26
# @Author : Dong
# @File   : position_init.py

import numpy as np
from app.plugins.transfer_function.trans_function import trans_func as tf


def position_inti(agents_number, dim, ub, lb, type='salp'):
	positions_array = None
	if type == 'salp':
		positions_array = np.random.uniform(lb, ub, (agents_number, dim)) * (ub - lb) + lb

	return positions_array


if __name__=='__main__':
	postion = position_inti(3, 5, 1, 0)
	print(postion)
	# print('==='*20)
	# print(tf(postion, type=3))