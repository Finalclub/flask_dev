#!/user/bin/env/ python
# -*- coding:utf-8 -*-

# @Time   : 2021/7/12 23:19
# @Author : Dong
# @File   : SQL.py

import pymysql
from flask import current_app


class SQLHelper(object):
	@staticmethod
	def open():
		try:
			pool = current_app.config['PYMYSQL_POOL']
		except:
			from dbutils.pooled_db import PooledDB
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
					host='127.0.0.1',
					port=3306,
					user='luka',
					password='ps897570831',
					database='dev_study',
					charset='utf8'
				)
		conn = pool.connection()
		cursor = conn.cursor(cursor=pymysql.cursors.DictCursor)
		return conn, cursor

	@staticmethod
	def close(conn, cursor):
		conn.commit()
		cursor.close()
		conn.close()

	@classmethod
	def fetch_one(cls, sql, **args):
		conn, cursor = cls.open()
		cursor.execute(sql, **args)
		obj = cursor.fetchone()
		cls.close(conn, cursor)
		return obj

	@classmethod
	def fetch_all(cls, sql, **args):
		conn, cursor = cls.open()
		cursor.execute(sql, **args)
		obj = cursor.fetchall()
		cls.close(conn, cursor)
		return obj


if __name__ == '__main__':
	obj = SQLHelper.fetch_one("select * from user")
	print(obj)
