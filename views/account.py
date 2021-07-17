#!/user/bin/env/ python
# -*- coding:utf-8 -*-

# @Time   : 2021/7/11 15:51
# @Author : Dong
# @File   : account.py

from flask import Blueprint,render_template, request, session, redirect, flash
from flask_login import login_user, login_required, logout_user, current_user
from utils.SQL import SQLHelper
from wtforms import Form
from wtforms import validators
from wtforms import widgets
from wtforms.fields import simple, html5

ac = Blueprint('account', __name__)


class LoginForm(Form):
	user = simple.StringField(
		label='用户名',
		validators=[
			validators.DataRequired(message='用户名不能为空.'),
			validators.Length(min=6, max=18, message='用户名长度必须大于%(min)d且小于%(max)d')
		],
		widget=widgets.TextInput(),
		render_kw={'class': 'form-control', 'placeholder': '请输入用户名'}

	)
	pwd = simple.PasswordField(
		label='密码',
		validators=[
			validators.DataRequired(message='密码不能为空.'),
			validators.Length(min=8, message='密码长度必须大于%(min)d'),
			validators.Regexp(regex="^(?=.*[a-z])(?=.*[A-Z])(?=.*\d)(?=.*[$@$!%*?&])[A-Za-z\d$@$!%*?&]{8,}",
							  message='密码至少8个字符，至少1个大写字母，1个小写字母，1个数字和1个特殊字符')

		],
		widget=widgets.PasswordInput(),
		render_kw={'class': 'form-control', 'placeholder': '请输入密码'})


class RegisterForm(Form):
	user = simple.StringField(
		label='用户名',
		validators=[
			validators.DataRequired()
		],
		widget=widgets.TextInput(),
		render_kw={'class': 'form-control', 'placeholder': '请输入用户名'},
	)

	pwd = simple.PasswordField(
		label='密码',
		validators=[
			validators.DataRequired(message='密码不能为空.')
		],
		widget=widgets.PasswordInput(),
		render_kw={'class': 'form-control', 'placeholder': '请输入密码'}
	)

	pwd_confirm = simple.PasswordField(
		label='重复密码',
		validators=[
			validators.DataRequired(message='重复密码不能为空.'),
			validators.EqualTo('pwd', message="两次密码输入不一致")
		],
		widget=widgets.PasswordInput(),
		render_kw={'class': 'form-control', 'placeholder': '请再次输入密码'}
	)

	email = html5.EmailField(
		label='邮箱',
		validators=[
			validators.DataRequired(message='邮箱不能为空.'),
			validators.Email(message='邮箱格式错误')
		],
		widget=widgets.TextInput(input_type='email'),
		render_kw={'class': 'form-control', 'placeholder': '请输入邮箱'}
	)

	def __init__(self, *args, **kwargs):
		super(RegisterForm, self).__init__(*args, **kwargs)
		# self.favor.choices = ((1, '篮球'), (2, '足球'), (3, '羽毛球'))

	def validate_user(self, field):
		"""
		钩子函数
		自定义pwd_confirm字段规则，例：与pwd字段是否一致
		:param field:
		:return:
		"""
		obj = SQLHelper.fetch_one("select id from user where name=%s", [field.data, ])
		# print(field.data)
		if obj:
			raise validators.StopValidation('用户名已存在')

	def validate_pwd_confirm(self, field):
		"""
		钩子函数
		自定义pwd_confirm字段规则，例：与pwd字段是否一致
		:param field:
		:return:
		"""
		# 最开始初始化时，self.data中已经有所有的值
		# field.data 当前name传过来的值
		# self.data 当前传过来的所有值
		if field.data != self.data['pwd']:
			# raise validators.ValidationError("密码不一致") # 继续后续验证
			raise validators.StopValidation("密码不一致")  # 不再继续后续验证


@ac.before_request
def check_login():
	if request.path == '/login':
		return
	user = session.get('user')
	print(user)
	if not user:
		return redirect('/login')


@ac.route('/')
def default_page():
	return redirect('/login')


@ac.route('/login', methods=['GET', 'POST'])
def login():
	if request.method == 'GET':
		form = LoginForm()
		return render_template('login.html', form=form)
	form = LoginForm(formdata=request.form)
	# validate
	if form.validate():
		print('用户提交数据通过校验，提交的值为', form.data)
	else:
		print(form.errors)
	# return render_template('login.html', form=form)
	obj = SQLHelper.fetch_one("select * from user where name=%(user)s and pwd=%(pwd)s", form.data)
	# print(obj)

	if obj:
		session.permanent = True
		session['user'] = {'id': obj['id'], 'name': obj['name']}
		print(session.get('user'))
		# current_app.carrier.login(form.data['user'])
		# login_user(session.get('user'))
		flash('成功登录')
		return redirect('/index')
	else:

		return render_template('login.html', form=form)


# @ac.route('/login', methods=['GET', 'POST'])
# def login():
# 	if request.method == 'GET':
# 		return render_template('login_test.html')
# 	user = request.form.get('user')
# 	pwd = request.form.get('pwd')
# 	obj = SQLHelper.fetch_one("select * from user where name=%s and pwd=%s", [user, pwd, ])
# 	if obj:
# 		session.permanent = True
# 		session['user_info'] = {'id':obj['id'], 'name':user}
# 		return redirect('/index')
# 用户登录状态校验


@ac.route('/logout')
def logout():
	# current_app.carrier.logout()
	return redirect('/login')


@ac.route('/register', methods=['GET', 'POST'])
def register():
	if request.method == 'GET':
		form = RegisterForm()
		return render_template('registration.html', form=form)
	form = RegisterForm(formdata=request.form)
	if form.validate():
		print(form.data)
		SQLHelper.write_db("insert into user (name, pwd, email) values (%(user)s, %(pwd_confirm)s, %(email)s)", form.data)
		return redirect('/login')
	else:
		print(form.errors)
		flash('请输入正确格式的信息或用户已存在')
		return render_template('registration.html', form=form)
