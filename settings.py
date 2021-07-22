#!/user/bin/env/ python
# -*- coding:utf-8 -*-

# @Time   : 2021/7/13 16:09
# @Author : Dong
# @File   : settings.py
# import pymysql
import os
from redis import Redis
from datetime import timedelta
# from dbutils.pooled_db import PooledDB


class BaseConfig(object):
	DEBUG = True
	# coockie 加密设定
	SECRET_KEY = 'asdfasdhjfjkghabsjdgfh'
	PERMANET_SESSION_LIFETIME = timedelta(minutes=2400)
	SESSION_REFRESH_ESCH_REQUEST = True
	SESSION_TYPE = 'redis'
	# REDIS_URL = 'redis://:luka@127.0.0.1:6379/'


class ProductionConfig(BaseConfig):
	DATABASE_URI = ''


class DevelopmentConfig(BaseConfig):
	SESSION_REDIS = Redis(host='localhost', port='6379', password='luka', socket_keepalive=True)
	# PYMYSQL_POOL = PooledDB(
	# 	creator='',
	# 	maxconnections=6,
	# 	mincached=2,
	# 	maxcached=5,
	# 	maxshared=3,
	# 	blocking=True,
	# 	maxusage=None,
	# 	setsession=[],
	# 	ping=0,
	# 	host='127.0.0.1',
	# 	port=3306,
	# 	user='luka',
	# 	password='ps897570831',
	# 	database='dev_study',
	# 	charset='utf8mb4'
	# )
	BASEURL = 'mysql+{}://{}:{}@localhost:3306/dev_study?charset={}'
	SQLALCHEMY_DATABASE_URI = BASEURL.format('mysqlconnector', 'luka', 'ps897570831', 'utf8mb4')
	SQLALCHEMY_TRACK_MODIFICATIONS = False
	UPLOAD_FILES_DEST = os.path.join(os.path.dirname(__file__), 'uploads')



