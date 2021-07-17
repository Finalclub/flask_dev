#!/user/bin/env/ python
# -*- coding:utf-8 -*-

# @Time   : 2021/7/14 22:14
# @Author : Dong
# @File   : home.py

from flask import Blueprint,render_template, request, session, redirect, current_app

hm = Blueprint('home', __name__)


@hm.before_request
def check_login():
	if request.path == '/login':
		return
	user = session.get('user')
	print(user)
	if not user:
		return redirect('/login')


@hm.route('/index', methods=['GET', 'POST'])
def index():
	return render_template('index.html')


@hm.context_processor
def context():
	user = session.get('user')['name']
	return dict(current_user=user)