#!/user/bin/env/ python
# -*- coding:utf-8 -*-

# @Time   : 2021/7/19 20:00
# @Author : Dong
# @File   : fasddfags.py

from app.plugins.LSH.readimage import ReadImage
import json
import numpy as np

ri = ReadImage()

path = 'sdfasdfasfa\nevsdggdsads\iris.csv'

c = [-10, -5, 0, 5, 3, 10, 15, -20, 25]

print(path)
print(c.index((min(c))))
print(c.index((max(c))))