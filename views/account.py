#!/user/bin/env/ python
# -*- coding:utf-8 -*-

# @Time   : 2021/7/11 15:51
# @Author : Dong
# @File   : account.py

from flask import Blueprint,render_template, request, session, redirect, current_app
from utils.SQL import SQLHelper

ac = Blueprint('account', __name__)


@ac.route('/login', methods=['GET', 'POST'])
def login():
	if request.method == 'GET':
		return render_template('login.html')
	user = request.form.get('loginname')
	pwd = request.form.get('password')
	print(user)
	print(pwd)
	obj = SQLHelper.fetch_one("select * from user where name=%s and pwd=%s", [user, pwd, ])
	if obj:
		session.permanent = True
		session['user_info'] = {'id': obj['id'], 'name': user}
		return redirect('/index')
	else:
		return redirect('login.html')