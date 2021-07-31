#!/user/bin/env/ python
# -*- coding:utf-8 -*-

# @Time   : 2021/7/18 21:49
# @Author : Dong
# @File   : forms.py

from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, BooleanField, SubmitField, TextAreaField
from wtforms import widgets
from wtforms.validators import ValidationError, DataRequired, Email, EqualTo, Length
from wtforms.fields import html5
from app.models import User


class LoginForm(FlaskForm):
	user = StringField(
		validators=[
			DataRequired(message='用户名不能为空.'),
			# Length(min=6, max=18, message='用户名长度必须大于%(min)d且小于%(max)d')
		],
		widget=widgets.TextInput(),
		render_kw={'class': 'form-control', 'placeholder': '请输入用户名'}
	)
	pwd = PasswordField(
		validators=[
			DataRequired(message='密码不能为空.'),
			# validators.Length(min=8, message='密码长度必须大于%(min)d'),
			# validators.Regexp(regex="^(?=.*[a-z])(?=.*[A-Z])(?=.*\d)(?=.*[$@$!%*?&])[A-Za-z\d$@$!%*?&]{8,}",
			# 				  message='密码至少8个字符，至少1个大写字母，1个小写字母，1个数字和1个特殊字符')

		],
		widget=widgets.PasswordInput(),
		render_kw={'class': 'form-control', 'placeholder': '请输入密码'})
	remember_me = BooleanField('记住我')
	submit = SubmitField('登录')


class RegisterForm(FlaskForm):
	user = StringField(
		validators=[
			DataRequired()
		],
		widget=widgets.TextInput(),
		render_kw={'class': 'form-control', 'placeholder': '请输入用户名'},
	)

	pwd = PasswordField(
		validators=[
			DataRequired(message='密码不能为空.')
		],
		widget=widgets.PasswordInput(),
		render_kw={'class': 'form-control', 'placeholder': '请输入密码'}
	)

	pwd_confirm = PasswordField(
		validators=[
			DataRequired(message='重复密码不能为空.'),
			EqualTo('pwd', message="两次密码输入不一致")
		],
		widget=widgets.PasswordInput(),
		render_kw={'class': 'form-control', 'placeholder': '请再次输入密码'}
	)

	email = html5.EmailField(
		validators=[
			DataRequired(message='邮箱不能为空.'),
			Email(message='邮箱格式错误')
		],
		widget=widgets.TextInput(),
		render_kw={'class': 'form-control', 'placeholder': '请输入邮箱'}
	)
	about_me = TextAreaField('About me', validators=[Length(min=0, max=140)],
							 render_kw={'class': 'form-control', 'placeholder': '请输入个人描述'})
	submit = SubmitField('提交')

	def __init__(self, *args, **kwargs):
		super(RegisterForm, self).__init__(*args, **kwargs)
		# self.favor.choices = ((1, '篮球'), (2, '足球'), (3, '羽毛球'))

	def validate_user(self, user):
		"""
		钩子函数
		自定义pwd_confirm字段规则，例：与pwd字段是否一致
		:param field:
		:return:
		"""
		user = User.query.filter_by(name=user.data).first()
		if user is not None:
			raise ValidationError('用户名已存在')


class EditForm(FlaskForm):
	user = StringField(
		validators=[
			DataRequired()
		],
		widget=widgets.TextInput(),
		render_kw={'class': 'form-control', 'placeholder': '请输入用户名'},
	)
	pwd = PasswordField(
		validators=[
			DataRequired(message='密码不能为空.')
		],
		widget=widgets.PasswordInput(),
		render_kw={'class': 'form-control', 'placeholder': '请输入密码'}
	)

	pwd_confirm = PasswordField(
		validators=[
			DataRequired(message='重复密码不能为空.'),
			EqualTo('pwd', message="两次密码输入不一致")
		],
		widget=widgets.PasswordInput(),
		render_kw={'class': 'form-control', 'placeholder': '请再次输入密码'}
	)
	email = html5.EmailField(
		validators=[
			DataRequired(message='邮箱不能为空.'),
			Email(message='邮箱格式错误')
		],
		widget=widgets.TextInput(),
		render_kw={'class': 'form-control', 'placeholder': '请输入邮箱'}
	)
	about_me = TextAreaField('About me',
							 validators=[Length(min=0, max=140)],
							 render_kw={'class': 'form-control', 'placeholder': '请输入个人描述'})
	submit = SubmitField('提交')

