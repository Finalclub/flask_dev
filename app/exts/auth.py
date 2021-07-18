#!/user/bin/env/ python
# -*- coding:utf-8 -*-

# @Time   : 2021/7/16 17:11
# @Author : Dong
# @File   : auth.py

from flask import request, session, redirect


class Auth(object):

	def __init__(self, app=None):
		self.app = app
		if app:
			self.init_app(app)

	def init_app(self, app):
		app.carrier = self
		self.app = app
		app.before_request(self.check_login)
		app.context_processor(self.context_processor)

	# 检验用户是否已经登录
	def check_login(self):
		if request.path == '/login':
			return
		user = session.get('user')
		if not user:
			return redirect('/login')
			# return render_template('login.html')
		print('check login')

	def login(self, data):
		session['user'] = data

	def context_processor(self):
		user = session.get('user')
		return dict(current_user=user)

	def logout(self):
		del session['user']