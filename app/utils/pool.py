#!/user/bin/env/ python
# -*- coding:utf-8 -*-

# @Time   : 2021/7/13 14:56
# @Author : Dong
# @File   : pool.py

import pymysql
from dbutils.pooled_db import PooledDB


def init_pool(app):
	pool = PooledDB(
		creator=pymysql,
		maxconnections=6,
		mincached=2,
		maxcached=5,
		maxshared=3,
		blocking=True,
		maxusage=None,
		setsession=[],
		ping=0,
		host=app.config['PYMYSQL_HOST'],
		port=app.config['PYMYSQL_PORT'],
		user=app.config['USER'],
		password=app.config['PASSWORADS'],
		database=app.config['DATABASE'],
		charset=app.config['CHARSET']
	)
	app.config['PYMYSQL_POOL'] = pool


def run_query(query):
	'''
	检测当前正在运行链接数是否小于最大连接数，如果不小于则：等待或报raise TooManyConnections异常
	否则优先去初始化时创建的链接中获取链接 SteadyDBConnection
	然后将SteadyDBConneciton对象封装到PooledDedicatedDBConnection中并返回。
	如果最开始创建的链接没有链接，则去创建一个SteadyDBConneciton对象，再封装到PooledDedicatedDBConnection中并返回。
	一旦链接关闭后，链接就返回到连接池让后续线程继续使用。
	:return: query search results
	'''
	conn = pool.connection()
	cursor = conn.cursor(cursor=pymysql.cursors.DictCursor)
	cursor.execute(query)
	result = cursor.fetchall()
	conn.close()
	return result


if __name__ == '__main__':
	query = '''
    	select * from dev_study.user
    	'''
	result = run_query(query=query)
	print(result)