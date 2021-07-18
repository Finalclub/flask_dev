#!/user/bin/env/ python
# -*- coding:utf-8 -*-

# @Time   : 2021/7/14 22:14
# @Author : Dong
# @File   : home.py

from flask import Blueprint,render_template, request, session, redirect, \
	g, url_for
from flask_login import current_user, login_required
from datetime import datetime

hm = Blueprint('home', __name__)


@hm.before_request
def check_login():
	if request.path == '/login':
		return
	user = session.get('user')
	print(user)
	if not user:
		return redirect('/login')


@hm.context_processor
@login_required
def context():
	user = session.get('user')['name']
	return dict(current_user=user)


@hm.route('/index', methods=['GET', 'POST'])
@login_required
def index():
	return render_template('index.html')


@hm.route('/index/user_infor/', methods=['GET', 'POST'])
@login_required
def user_infor():
	return render_template('user_infor.html')


@hm.route('/index/dataset_upload', methods=['GET', 'POST'])
@login_required
def dataset_upload():
	return render_template('user_infor.html')


@hm.route('/index/dataset_infor', methods=['GET', 'POST'])
@login_required
def dataset_infor():
	return render_template('user_infor.html')


@hm.route('/logout')
@login_required
def logout():
	del session['user']
	return redirect(url_for('account.login'))

