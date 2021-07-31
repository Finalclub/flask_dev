#!/user/bin/env/ python
# -*- coding:utf-8 -*-

# @Time   : 2021/7/18 20:23
# @Author : Dong
# @File   : models.py

from datetime import datetime
from flask_login import UserMixin
from werkzeug.security import generate_password_hash, check_password_hash
# import jwt
# import redis
# import rq
from app import db, login
# from app.search import add_to_index, remove_from_index, query_index


class User(UserMixin, db.Model):
	id = db.Column(db.Integer, primary_key=True, autoincrement=True, unique=True)
	name = db.Column(db.String(64), unique=True)
	pwd_hash = db.Column(db.String(128))
	email = db.Column(db.String(120))
	about_me = db.Column(db.TEXT)
	type = db.Column(db.Integer, db.ForeignKey('auth.id_auth'), default=0)
	last_seen = db.Column(db.DateTime, default=datetime.now)
	# auths = db.relationship('Auth', backref='auth', lazy='select')

	def set_password(self, pwd):
		self.pwd_hash = generate_password_hash(pwd)

	def check_password(self, pwd):
		return check_password_hash(self.pwd_hash, pwd)


class Auth(db.Model):
	id_auth = db.Column(db.Integer, primary_key=True, autoincrement=True, unique=True)
	name = db.Column(db.String(64), index=True, unique=True)
	created_date = db.Column(db.DateTime, default=datetime.now)
	# 记录修改时间
	modified_date = db.Column(db.DateTime, default=datetime.now, onupdate=datetime.now)


class Dataset(db.Model):
	id = db.Column(db.Integer, primary_key=True, autoincrement=True, unique=True)
	name = db.Column(db.String(64), index=True)
	sample_count = db.Column(db.Integer)
	feature_count = db.Column(db.Integer)
	format = db.Column(db.String(64))
	description = db.Column(db.TEXT)
	created_date = db.Column(db.DateTime, default=datetime.now)
	file_path = db.Column(db.String(120))
	dataset_type = db.Column(db.Integer, db.ForeignKey('datatype.id_dtype'))
	uri_id = db.Column(db.Integer, db.ForeignKey('user.id'))
	users = db.relationship('User', backref='users', lazy='select')


class BS(db.Model):
	id = db.Column(db.Integer, primary_key=True, autoincrement=True, unique=True)
	name = db.Column(db.String(64), index=True, unique=True)
	created_date = db.Column(db.DateTime, default=datetime.now)
	# 记录修改时间
	modified_date = db.Column(db.DateTime, default=datetime.now, onupdate=datetime.now)


class Classifer(db.Model):
	id = db.Column(db.Integer, primary_key=True, autoincrement=True, unique=True)
	name = db.Column(db.String(64), index=True, unique=True)
	created_date = db.Column(db.DateTime, default=datetime.now)
	# 记录修改时间
	modified_date = db.Column(db.DateTime, default=datetime.now, onupdate=datetime.now)


class Datatype(db.Model):
	id_dtype = db.Column(db.Integer, primary_key=True, autoincrement=True, unique=True)
	name = db.Column(db.String(64), index=True, unique=True)
	created_date = db.Column(db.DateTime, default=datetime.now)


@login.user_loader
def load_user(id):
	return User.query.get(int(id))