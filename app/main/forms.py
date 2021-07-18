#!/user/bin/env/ python
# -*- coding:utf-8 -*-

# @Time   : 2021/7/18 15:50
# @Author : Dong
# @File   : forms.py

from wtforms import Form, validators, widgets
from wtforms.fields import simple, html5
from app.utils.SQL import SQLHelper


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


class EditProfileForm(Form):
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