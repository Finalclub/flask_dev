#!/user/bin/env/ python
# -*- coding:utf-8 -*-

# @Time   : 2021/7/11 15:51
# @Author : Dong
# @File   : account.py

from flask import Blueprint, render_template, request, session, redirect, flash
from flask_login import current_user, login_user
from app.auth.forms import LoginForm, RegisterForm
from app.models import User
from app import db

ac = Blueprint('account', __name__)


# @ac.before_request
# def check_login():
# 	if request.path == '/login':
# 		return
# 	user = session.get('user')
# 	print(user)
# 	if not user:
# 		return redirect('/login')


@ac.route('/')
def default_page():
	return redirect('/login')


@ac.route('/login', methods=['GET', 'POST'])
def login():
	# if request == 'GET':
	# 	form = LoginForm()
	# 	return render_template('auth/../templates/login.html', form=form)
	if current_user.is_authenticated:
		return redirect('/index')
	form = LoginForm()

	if form.validate_on_submit():
		# print('用户提交数据通过校验，提交的值为', form.data)
		user = User.query.filter_by(name=form.user.data).first()
		if user is None or not user.check_password(form.pwd.data):
			flash('用户名或密码错误')
			return redirect('/login')
		login_user(user, remember=form.remember_me.data)
		session.permanent = True
		session['name'] = form.user.data
		return redirect('/index')
	else:
		print(1, form.user.errors)
		print(1, form.pwd.errors)
	return render_template('auth/login.html', form=form)


@ac.route('/register', methods=['GET', 'POST'])
def register():
	# if request.method == 'GET':
	# 	form = RegisterForm()
	# 	return render_template('auth/../templates/registration.html', form=form)
	if current_user.is_authenticated:
		return redirect('/index')
	form = RegisterForm()
	if form.validate_on_submit():
		# print(form.data)
		user = User(name=form.user.data, email=form.email.data, about_me=form.about_me.data)
		user.set_password(form.pwd.data)
		# print(user)
		# print(User.pwd_hash)
		db.session.add(user)
		db.session.commit()
		flash('注册成功！')
		return redirect('/login')
	else:
		print(2, form.user.errors)
		print(2, form.pwd.errors)
		print(2, form.pwd_confirm.errors)
		print(2, form.email.errors)
		print('something happened')

	return render_template('auth/registration.html', form=form)
