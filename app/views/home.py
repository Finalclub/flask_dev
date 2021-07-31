#!/user/bin/env/ python
# -*- coding:utf-8 -*-

# @Time   : 2021/7/14 22:14
# @Author : Dong
# @File   : home.py

from flask import Blueprint,render_template, request, redirect, \
	 url_for, jsonify, send_from_directory
from flask_login import current_user, login_required, logout_user
from werkzeug.utils import secure_filename
from werkzeug.datastructures import CombinedMultiDict
from werkzeug.security import generate_password_hash
from operator import truediv
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsOneClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import accuracy_score
from app.models import User, Dataset
from app.utils.SQL import SQLHelper
from app import db, celery
from app.auth.forms import EditForm
from app.main.forms import UploadForms, SelectDSForm, SelectDSFormSSA, \
	SelectDSFormSVM, SelectDSFormRF, SelectDSFormKNN
from app.plugins.LSH.readimage import ReadImage
from app.utils.common_use import get_algorithm_args, get_data, find_min_index, to_json, to_array
from app.plugins.HNEFS.HNEFS import NEFS
from app.plugins.SSA.quantum_slap_swarm_algorithm import QSSA
from datetime import datetime
import time
import numpy as np
import os


hm = Blueprint('home', __name__)

file_reader = ReadImage()

data_path = None
result_nefs = None
result_ssa = None
result_classifier = None
response_result = None


@celery.task(bind=True)
def backgroundNE_task(self, X, Y, k, w, L, n_bands):
	# 	初始化结果集
	if isinstance(X, str):
		X = to_array(X)
	if isinstance(Y, str):
		Y = to_array(Y)
	S = list()
	D = X.shape[-1]
	# len_s = len(S)
	band_index_list = [x for x in range(D)]
	# while len(S) < n_bands:
	tic  = time.time()
	for i in range(n_bands):
		# len_s = len(S)
		set_for_search = set(band_index_list) - set(S)
		ne_rank_dict = dict()
		# tic = time()
		for band in set_for_search:
			nefs_band = S + [band]
			# print('nefs_band', nefs_band)
			# 计算并记录NE
			ne_rank_dict[str(nefs_band)] = NEFS(inputarray=X[:, nefs_band], inputgt=Y, k=k, w=w, L=L)
		# 获取最小NE值的key
		key_min = find_min_index(ne_rank_dict)
		key_min = key_min.replace('[', '').replace(']', '')
		# 更新S
		S = list(set(key_min.split(',') + S))
		S = [int(x) for x in S]
		S = list(set(S))
		message = 'progressing'
		self.update_state(state='PROGRESS', meta={'current': i, 'total': n_bands, 'status': message})
		time.sleep(1)
	toc = time.time()
	# int_list = [int(x) for x in S]
	result = 'Selected: {}  Cost:{}s'.format(S, toc - tic)
	return {'current': len(S), 'total': n_bands, 'status': '任务完成', 'result': result}


@celery.task(bind=True)
def backgroundSSA_task(self, dataset, labels, population, max_iter, times):
	if isinstance(dataset, str):
		X = to_array(dataset)
	if isinstance(labels, str):
		Y = to_array(labels)
	fitness_list = list()
	index_list = list()
	tic = time.time()
	for i in range(times):
		selected, best_fitness = QSSA(n_population=population, max_iter=max_iter, dataset=X, labels=Y, trans_type=7, ub=1, lb=0)
		index_list.append(selected)
		fitness_list.append(best_fitness)
		message = 'progressing'
		self.update_state(state='PROGRESS',
							meta={'current': i, 'total': times, 'status': message})
		time.sleep(1)
	toc = time.time()
	avg = np.mean(fitness_list)
	std = np.std(fitness_list, ddof=1)
	worst = max(fitness_list)
	best = min(fitness_list)
	best_index = index_list[fitness_list.index(best)]
	results = 'Avg:{}   Std:{}   Best:{}   Worst:{}   Selected: {}  Cost:{}s'.format(avg, std, best, worst,
																					best_index, toc - tic)
	return {'current': 100, 'total': 100, 'status': 'Task completed!', 'result': results}


@celery.task(bind=True)
def backgroundsvm_task(self, X, y, test_size, subset: list, kernel='rbf'):
	if isinstance(X, str):
		X = to_array(X)
	if isinstance(y, str):
		y = to_array(y)
	if subset != list():
		subset = [int(x) for x in subset]
		X = X[:, subset]
	sfs = SVC(kernel=kernel, probability=True)
	tic = time.time()
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, shuffle=True, random_state=0)
	self.update_state(state='PROGRESS', meta={'current': 1, 'total': 4, 'status': 'processing'})
	clf = OneVsOneClassifier(sfs).fit(X_train, y_train)

	y_pre = clf.predict(X_test)

	labels = list(set(y.tolist()))
	# print(labels)

	cf = confusion_matrix(y_true=y_test, y_pred=y_pre, labels=labels)

	list_diag = np.diag(cf)

	list_raw_sum = np.sum(cf, axis=1)

	each_acc = np.nan_to_num(truediv(list_diag, list_raw_sum))
	each_acc_list = each_acc.tolist()
	# 返回每个类的分类acc，类型array
	self.update_state(state='PROGRESS', meta={'current': 2, 'total': 4, 'status': 'processing'})

	oa = accuracy_score(y_true=y_test, y_pred=y_pre)
	self.update_state(state='PROGRESS', meta={'current': 3, 'total': 4, 'status': 'processing'})

	kc = cohen_kappa_score(y1=y_test, y2=y_pre, labels=labels)

	each_acc_result = str(each_acc_list).replace('[', '').replace(']', '')
	toc = time.time()
	results = 'OA:{}  KC:{}  Accuracy of each label(default index): {}  Cost:{}s'.format(oa, kc, each_acc_result, toc - tic)
	return {'current': 1, 'total': 1, 'status': 'Task completed!', 'result': results}


@celery.task(bind=True)
def backgroundrf_task(self, X, y, test_size, subset: list, n_estimators, min_samples_split, min_samples_leaf):
	if isinstance(X, str):
		X = to_array(X)
	if isinstance(y, str):
		y = to_array(y)
	if subset != list():
		subset = [int(x) for x in subset]
		X = X[:, subset]
	sfs = RandomForestClassifier(n_estimators=n_estimators,
								 min_samples_split=min_samples_split,
								 min_samples_leaf=min_samples_leaf)

	tic = time.time()
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, shuffle=True, random_state=0)
	self.update_state(state='PROGRESS', meta={'current': 1, 'total': 4, 'status': 'processing'})
	clf = OneVsOneClassifier(sfs).fit(X_train, y_train)

	y_pre = clf.predict(X_test)

	labels = list(set(y.tolist()))

	cf = confusion_matrix(y_true=y_test, y_pred=y_pre, labels=labels)

	list_diag = np.diag(cf)

	list_raw_sum = np.sum(cf, axis=1)

	each_acc = np.nan_to_num(truediv(list_diag, list_raw_sum))
	each_acc_list = each_acc.tolist()
	# 返回每个类的分类acc，类型array
	self.update_state(state='PROGRESS', meta={'current': 2, 'total': 4, 'status': 'processing'})
	oa = accuracy_score(y_true=y_test, y_pred=y_pre)
	self.update_state(state='PROGRESS', meta={'current': 3, 'total': 4, 'status': 'processing'})

	kc = cohen_kappa_score(y1=y_test, y2=y_pre, labels=labels)

	each_acc_result = str(each_acc_list).replace('[', '').replace(']', '')
	toc = time.time()
	results = 'OA:{}  KC:{}  Accuracy of each label(default index): {}  Cost:{}s'.format(oa, kc, each_acc_result, toc - tic)
	return {'current': 1, 'total': 1, 'status': 'Task completed!', 'result': results}


@celery.task(bind=True)
def backgroundknn_task(self, X, y, test_size, subset: list, n_neighbors):
	if isinstance(X, str):
		X = to_array(X)
	if isinstance(y, str):
		y = to_array(y)
	if subset != list():
		subset = [int(x) for x in subset]
		X = X[:, subset]
	tic = time.time()
	sfs = KNeighborsClassifier(n_neighbors=n_neighbors)
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, shuffle=True, random_state=0)
	self.update_state(state='PROGRESS', meta={'current': 1, 'total': 4, 'status': 'processing'})
	clf = OneVsOneClassifier(sfs).fit(X_train, y_train)

	y_pre = clf.predict(X_test)

	labels = list(set(y.tolist()))

	cf = confusion_matrix(y_true=y_test, y_pred=y_pre, labels=labels)

	list_diag = np.diag(cf)

	list_raw_sum = np.sum(cf, axis=1)

	each_acc = np.nan_to_num(truediv(list_diag, list_raw_sum))
	each_acc_list = each_acc.tolist()
	# 返回每个类的分类acc，类型array
	self.update_state(state='PROGRESS', meta={'current': 2, 'total': 4, 'status': 'processing'})
	oa = accuracy_score(y_true=y_test, y_pred=y_pre)
	self.update_state(state='PROGRESS', meta={'current': 3, 'total': 4, 'status': 'processing'})
	kc = cohen_kappa_score(y1=y_test, y2=y_pre, labels=labels)
	each_acc_result = str(each_acc_list).replace('[', '').replace(']', '')
	toc = time.time()
	results = 'OA:{}  KC:{}  Accuracy of each label(default index): {}  Cost:{}s'.format(oa, kc, each_acc_result,
																						 toc - tic)
	return {'current': 1, 'total': 1, 'status': 'Task completed!', 'result': results}


@hm.before_request
def before_request():
	if current_user.is_authenticated:
		current_user.last_seen = datetime.now()
		db.session.commit()


@hm.route('/index', methods=['GET', 'POST'])
@login_required
def index():
	return render_template('index.html')


@hm.route('/edit_profile', methods=['GET', 'POST'])
@login_required
def edit_profile():
	form = EditForm()
	if form.validate_on_submit():
		name = form.user.data
		pwd_hash = generate_password_hash(form.pwd.data)
		email = form.email.data
		about_me = form.about_me.data
		query = '''update user set name='{}', pwd_hash='{}', email='{}', about_me='{}' where id={}'''
		query = query.format(name, pwd_hash, email, about_me, current_user.id)
		print(query)
		SQLHelper.write_db(query)
		return redirect(url_for('home.user_infor', user=name))
	elif request.method == 'GET':
		form.user.data = current_user.name
		form.about_me.data = current_user.about_me
		form.email.data = current_user.email
	return render_template('edit_page.html', form=form)


@hm.route('/index/user_infor/<user>', methods=['GET', 'POST'])
@login_required
def user_infor(user):
	name = User.query.filter_by(name=user).first()
	return render_template('user_infor.html', user=name)


@hm.route('/index/dataset_upload', methods=['GET', 'POST'])
@login_required
def dataset_upload():
	form = UploadForms(CombinedMultiDict([request.form, request.files]))
	if request.method == 'POST':
		if form.validate():
			dataset_check = Dataset.query.filter_by(name=form.file.data.filename).first()
			if dataset_check is None:
				file_handler = form.file.data
				server_path = os.path.join(os.path.split(os.path.dirname(__file__))[0], 'uploads')
				abs_path = os.path.join(server_path, file_handler.filename)
				file_handler.save(abs_path)
				array = file_reader.funReadMat(abs_path)
				array_shape = array.shape
				db_info_saver = Dataset(name=file_handler.filename,
										sample_count=array_shape[0],
										feature_count=array_shape[1],
										format=secure_filename(file_handler.filename.split('.')[-1]),
										description=form.desc.data,
										created_date=datetime.now(),
										file_path=abs_path,
										dataset_type=form.types.data,
										uri_id=current_user.id)
				db.session.add(db_info_saver)
				db.session.commit()
			return '文件上传成功'
		else:
			return form.errors

	return render_template('dataset_upload.html', form=form)


@hm.route('/index/dataset_infor', methods=['GET', 'POST'])
@login_required
def dataset_infor():
	query = '''
	select sample_count, feature_count, format, file_path, created_date,
	description, dataset_type from dataset order by id
	'''
	query_result = SQLHelper.fetch_all(query)
	return render_template('dataset_infor.html', query_result=query_result)


@hm.route('/index/NEFS', methods=['GET', 'POST'])
@login_required
def nefs():
	form = SelectDSForm()
	if request.method == 'POST':
		if form.validate_on_submit():
			global data_path
			db_path, lb_path, k, w, L, n_bands = get_algorithm_args(form, type='ne')
			X, y = get_data(db_path[0], lb_path[0])
			global result_nefs
			result_nefs = [X, y, k, w, L, n_bands]
			data_path = db_path
			return '成功提交，可以执行算法'
	return render_template('nebs.html', form=form)


@hm.route('/NEFS_task', methods=['POST'])
@login_required
def nefstask():
	global result_nefs
	task = backgroundNE_task.apply_async(
		args=[
			to_json(result_nefs[0]),
			to_json(result_nefs[1]),
			result_nefs[2],
			result_nefs[3],
			result_nefs[4],
			result_nefs[5]])
	return jsonify({}), 202, {'Location': url_for('home.nefsstatus', task_id=task.id)}


@hm.route('/NEFS_status/<task_id>')
@login_required
def nefsstatus(task_id):
	task = backgroundNE_task.AsyncResult(task_id)
	if task.state == 'PENDING':
		response = {
			'state': task.state,
			'current': 0,
			'total': 1,
			'status': 'Pending...'
		}
	elif task.state != 'FAILURE':
		response = {
			'state': task.state,
			'current': task.info.get('current', 0),
			'total': task.info.get('total', 1),
			'status': task.info.get('status', '')
		}
		if 'result' in task.info:
			global response_result
			response['result'] = task.info['result']
			response_result = task.info['result']
	else:
		# 后端出现问题
		response = {'state': task.state, 'current': 1, 'total': 1, 'status': str(task.info)}
		# 报错的具体异常
	return jsonify(response)


@hm.route('/index/QSSA', methods=['GET', 'POST'])
@login_required
def qssafs():
	form = SelectDSFormSSA()
	if request.method == 'POST':
		if form.validate_on_submit():
			global result_ssa
			global data_path
			db_path, lb_path, pop, max_iter, times = get_algorithm_args(form, type='ssa')
			X, y = get_data(db_path[0], lb_path[0])
			result_ssa = [X, y, pop, max_iter, times]
			data_path = db_path
			return '成功提交，可以执行算法'
	return render_template('qssa.html', form=form)


@hm.route('/QSSA_task', methods=['POST'])
def qssatask():
	global result_ssa
	task = backgroundSSA_task.apply_async(
		args=[
			to_json(result_ssa[0]),
			to_json(result_ssa[1]),
			result_ssa[2],
			result_ssa[3],
			result_ssa[4]
		]
	)
	return jsonify({}), 202, {'Location': url_for('home.qssastatus', task_id=task.id)}


@hm.route('/QSSA_status/<task_id>')
def qssastatus(task_id):
	task = backgroundSSA_task.AsyncResult(task_id)
	if task.state == 'PENDING':
		response = {
			'state': task.state,
			'current': 0,
			'total': 1,
			'status': 'Pending...'
		}
	elif task.state != 'FAILURE':
		response = {
			'state': task.state,
			'current': task.info.get('current', 0),
			'total': task.info.get('total', 1),
			'status': task.info.get('status', '')
		}
		if 'result' in task.info:
			global response_result
			response['result'] = task.info['result']
			response_result = task.info['result']
	else:
		# something went wrong in the background job
		response = {
			'state': task.state,
			'current': 1,
			'total': 1,
			'status': str(task.info),  # this is the exception raised
		}
	return jsonify(response)


@hm.route('/download_fs')
@login_required
def download_fs():
	global response_result
	if response_result is not None:
		index = response_result.split('[')[-1].split(']')[0]
		index_list = [int(x) for x in index.split(',')]
		global data_path
		X = file_reader.funReadMat(data_path[0])
		X_reduction = X[:, index_list]
		server_path = os.path.join(os.path.split(os.path.dirname(__file__))[0], 'download')
		file_name = os.path.basename(data_path[0]).split('.')[0]
		newfile_name = '{}_{}_redution.csv'.format(file_name, len(index_list))
		abs_path = os.path.join(server_path, newfile_name)
		np.savetxt(abs_path, X_reduction, delimiter=',')
		return send_from_directory(server_path, newfile_name, as_attachment=True)


@hm.route('/download_classifer')
@login_required
def download_classifer():
	global response_result
	if response_result is not None:
		global data_path
		server_path = os.path.join(os.path.split(os.path.dirname(__file__))[0], 'download')
		file_name = os.path.basename(data_path).split('.')[0]
		newfile_name = '{}_classifier_result.txt'.format(file_name)
		abs_path = os.path.join(server_path, newfile_name)
		with open(abs_path, 'w') as file:
			file.writelines(response_result)
		return send_from_directory(server_path, newfile_name, as_attachment=True)


@hm.route('/classifer_svm', methods=['GET', 'POST'])
@login_required
def svm():
	form = SelectDSFormSVM()
	if request.method == 'POST':
		if form.validate_on_submit():
			global result_classifier
			global data_path
			db_path, lb_path, test_set, subset = get_algorithm_args(form, type='svm')
			X, y = get_data(db_path[0], lb_path[0])
			subset = subset.replace('[', '').replace(']', '')
			subset_list = subset.split(',')
			if subset_list[0] == '':
				subset_list = list()
			result_classifier = [X, y, test_set, subset_list]
			data_path = db_path[0]
			return '成功提交，可以执行算法'
		else:
			print(form.errors)
	return render_template('svm.html', form=form)


@hm.route('/svm_task', methods=['POST'])
@login_required
def svmtask():
	global result_classifier
	task = backgroundsvm_task.apply_async(
		args=[
			to_json(result_classifier[0]),
			to_json(result_classifier[1]),
			result_classifier[2],
			result_classifier[3]
		]
	)
	return jsonify({}), 202, {'Location': url_for('home.svmstatus', task_id=task.id)}


@hm.route('/svm_status/<task_id>')
@login_required
def svmstatus(task_id):
	task = backgroundsvm_task.AsyncResult(task_id)
	if task.state == 'PENDING':
		response = {
			'state': task.state,
			'current': 0,
			'total': 1,
			'status': 'Pending...'
		}
	elif task.state != 'FAILURE':
		response = {
			'state': task.state,
			'current': task.info.get('current', 0),
			'total': task.info.get('total', 1),
			'status': task.info.get('status', '')
		}
		if 'result' in task.info:
			response['result'] = task.info['result']
			global response_result
			response_result = task.info['result']
	else:
		# something went wrong in the background job
		response = {
			'state': task.state,
			'current': 1,
			'total': 1,
			'status': str(task.info),  # this is the exception raised
		}
	return jsonify(response)


@hm.route('/classifer_rf', methods=['GET', 'POST'])
@login_required
def rf():
	form = SelectDSFormRF()
	if request.method == 'POST':
		if form.validate_on_submit():
			global result_classifier
			global data_path
			db_path, lb_path, test_set, subset, ntree, nsp, lsp = get_algorithm_args(form, type='rf')
			X, y = get_data(db_path[0], lb_path[0])
			subset = subset.replace('[', '').replace(']', '')
			subset_list = subset.split(',')
			if subset_list[0] == '':
				subset_list = list()
			result_classifier = [X, y, test_set, subset_list, ntree, nsp, lsp]
			data_path = db_path[0]
			return '成功提交，可以执行算法'
	return render_template('rf.html', form=form)


@hm.route('/rf_task', methods=['POST'])
@login_required
def rftask():
	global result_classifier
	task = backgroundrf_task.apply_async(
		args=[
			to_json(result_classifier[0]),
			to_json(result_classifier[1]),
			result_classifier[2],
			result_classifier[3],
			result_classifier[4],
			result_classifier[5],
			result_classifier[6]
		]
	)
	return jsonify({}), 202, {'Location': url_for('home.rfstatus', task_id=task.id)}


@hm.route('/rf_status/<task_id>')
@login_required
def rfstatus(task_id):
	task = backgroundrf_task.AsyncResult(task_id)
	if task.state == 'PENDING':
		response = {
			'state': task.state,
			'current': 0,
			'total': 1,
			'status': 'Pending...'
		}
	elif task.state != 'FAILURE':
		response = {
			'state': task.state,
			'current': task.info.get('current', 0),
			'total': task.info.get('total', 1),
			'status': task.info.get('status', '')
		}
		if 'result' in task.info:
			response['result'] = task.info['result']
			global response_result
			response_result = task.info['result']
	else:
		# something went wrong in the background job
		response = {
			'state': task.state,
			'current': 1,
			'total': 1,
			'status': str(task.info),  # this is the exception raised
		}
	return jsonify(response)


@hm.route('/classifer_knn', methods=['GET', 'POST'])
@login_required
def knn():
	form = SelectDSFormKNN()
	if request.method == 'POST':
		if form.validate_on_submit():
			global result_classifier
			global data_path
			db_path, lb_path, test_set, subset, k = get_algorithm_args(form, type='knn')
			X, y = get_data(db_path[0], lb_path[0])
			subset = subset.replace('[', '').replace(']', '')
			subset_list = subset.split(',')
			if subset_list[0] == '':
				subset_list = list()
			result_classifier = [X, y, test_set, subset_list, k]
			data_path = db_path[0]
			return '成功提交，可以执行算法'
	return render_template('knn.html', form=form)


@hm.route('/knn_task', methods=['POST'])
@login_required
def knntask():
	global result_classifier
	task = backgroundknn_task.apply_async(
		args=[
			to_json(result_classifier[0]),
			to_json(result_classifier[1]),
			result_classifier[2],
			result_classifier[3],
			result_classifier[4]
		]
	)
	return jsonify({}), 202, {'Location': url_for('home.knnstatus', task_id=task.id)}


@hm.route('/knn_status/<task_id>')
@login_required
def knnstatus(task_id):
	task = backgroundknn_task.AsyncResult(task_id)
	if task.state == 'PENDING':
		response = {
			'state': task.state,
			'current': 0,
			'total': 1,
			'status': 'Pending...'
		}
	elif task.state != 'FAILURE':
		response = {
			'state': task.state,
			'current': task.info.get('current', 0),
			'total': task.info.get('total', 1),
			'status': task.info.get('status', '')
		}
		if 'result' in task.info:
			response['result'] = task.info['result']
			global response_result
			response_result = task.info['result']

	else:
		# something went wrong in the background job
		response = {
			'state': task.state,
			'current': 1,
			'total': 1,
			'status': str(task.info),  # this is the exception raised
		}
	return jsonify(response)


@hm.route('/logout')
@login_required
def logout():
	logout_user()
	return redirect(url_for('account.login'))


