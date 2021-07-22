#!/usr/bin/env python3
# _*_ coding:utf-8 _*_
# @Version : 1.0
# @Time    : 2019/4/15 14:08
# @Author  : zhaidongchang
# @File    : hashtable.py


class HashTable(object):
	"""Initalize for HashTable"""
	def __init__(self, index):
		self.index = index
		self.buckets = {}