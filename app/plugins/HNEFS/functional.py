#!/user/bin/env/ python
# -*- coding:utf-8 -*-

# @Time   : 2021/4/20 17:18
# @Author : Dong
# @File   : functional.py

import time

tic = time.time()

a = ''
a_list = a.split(',')


print(a_list)

print(len(a_list))

toc = time.time()

print('{}s'.format(toc - tic))