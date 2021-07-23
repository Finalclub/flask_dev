#!/user/bin/env/ python
# -*- coding:utf-8 -*-

# @Time   : 2021/7/18 15:50
# @Author : Dong
# @File   : forms.py

from flask_wtf import FlaskForm
from wtforms.fields import core
from flask_wtf.file import FileField, FileAllowed, FileRequired
from wtforms import IntegerField, StringField, SubmitField, FloatField
from wtforms.validators import InputRequired
from app.utils.SQL import SQLHelper


class UploadForms(FlaskForm):
	file = FileField(u'上传数据集', validators=[FileAllowed(['csv', 'mat'], u'文件格式异常'), FileRequired()])
	desc = StringField(validators=[InputRequired('填写描述信息')])
	types = core.RadioField(
		choices=(
			(1, '数据集'),
			(2, '类标')
		),
		coerce=int
	)
	submit = SubmitField(u'上传')


class SelectDSForm(FlaskForm):
	dataset = core.SelectField(
		choices=SQLHelper.fetch_all('select id, name from dataset where dataset_type=1')
	)
	labels = core.SelectField(
		choices=SQLHelper.fetch_all('select id, name from dataset where dataset_type=2')
	)

	k = IntegerField(validators=[InputRequired('填写参数k')], default=20)
	w = IntegerField(validators=[InputRequired('填写参数w')], default=5)
	L = IntegerField(validators=[InputRequired('填写参数L')], default=30)
	n_bands = IntegerField(validators=[InputRequired('选取特征个数')])
	submit = SubmitField(u'上传')

	def __init__(self):
		super(SelectDSForm, self).__init__()
		self.labels.choices = SQLHelper.fetch_all('select id, name from dataset where dataset_type=1')
		self.labels.choices = SQLHelper.fetch_all('select id, name from dataset where dataset_type=2')


class SelectDSFormSSA(FlaskForm):
	dataset = core.SelectField(
		choices=SQLHelper.fetch_all('select id, name from dataset where dataset_type=1')
	)
	labels = core.SelectField(
		choices=SQLHelper.fetch_all('select id, name from dataset where dataset_type=2')
	)

	pop = IntegerField(validators=[InputRequired('填写种群个数')], default=50)
	max_iter = IntegerField(validators=[InputRequired('填写最大迭代次数')], default=100)
	times = IntegerField(validators=[InputRequired('填写实验次数')], default=20)
	submit = SubmitField(u'上传')

	def __init__(self):
		super(SelectDSFormSSA, self).__init__()
		self.labels.choices = SQLHelper.fetch_all('select id, name from dataset where dataset_type=1')
		self.labels.choices = SQLHelper.fetch_all('select id, name from dataset where dataset_type=2')


class SelectDSFormSVM(FlaskForm):
	dataset = core.SelectField(
		choices=SQLHelper.fetch_all('select id, name from dataset where dataset_type=1')
	)
	labels = core.SelectField(
		choices=SQLHelper.fetch_all('select id, name from dataset where dataset_type=2')
	)

	test_set = FloatField(validators=[InputRequired('填写测试集比例')], default=0.2)
	subset = StringField(validators=[InputRequired('填写特征子集')], default='')
	submit = SubmitField(u'上传')

	def __init__(self):
		super(SelectDSFormSVM, self).__init__()
		self.labels.choices = SQLHelper.fetch_all('select id, name from dataset where dataset_type=1')
		self.labels.choices = SQLHelper.fetch_all('select id, name from dataset where dataset_type=2')


class SelectDSFormRF(FlaskForm):
	dataset = core.SelectField(
		choices=SQLHelper.fetch_all('select id, name from dataset where dataset_type=1')
	)
	labels = core.SelectField(
		choices=SQLHelper.fetch_all('select id, name from dataset where dataset_type=2')
	)

	test_set = FloatField(validators=[InputRequired('填写测试集比例')], default=0.2)
	subset = StringField(validators=[InputRequired('填写特征子集')], default='')
	n_estimators = IntegerField(validators=[InputRequired('随机森林树的个数')], default=100)
	min_samples_split = IntegerField(validators=[InputRequired('分裂节点需要的最小样本数')], default=2)
	min_samples_leaf = IntegerField(validators=[InputRequired('叶子节点最小样本数')], default=1)
	submit = SubmitField(u'上传')

	def __init__(self):
		super(SelectDSFormRF, self).__init__()
		self.labels.choices = SQLHelper.fetch_all('select id, name from dataset where dataset_type=1')
		self.labels.choices = SQLHelper.fetch_all('select id, name from dataset where dataset_type=2')


class SelectDSFormKNN(FlaskForm):
	dataset = core.SelectField(
		choices=SQLHelper.fetch_all('select id, name from dataset where dataset_type=1')
	)
	labels = core.SelectField(
		choices=SQLHelper.fetch_all('select id, name from dataset where dataset_type=2')
	)

	test_set = FloatField(validators=[InputRequired('填写测试集比例')], default=0.2)
	subset = StringField(validators=[InputRequired('填写特征子集')], default='')
	n_neighbors = IntegerField(validators=[InputRequired('邻域个数')], default=20)
	submit = SubmitField(u'上传')

	def __init__(self):
		super(SelectDSFormKNN, self).__init__()
		self.labels.choices = SQLHelper.fetch_all('select id, name from dataset where dataset_type=1')
		self.labels.choices = SQLHelper.fetch_all('select id, name from dataset where dataset_type=2')


# debug
if __name__ == '__main__':
	pass