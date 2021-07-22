#!/user/bin/env/ python
# -*- coding:utf-8 -*-

# @Time   : 2021/7/14 22:14
# @Author : Dong
# @File   : home.py

from flask import Blueprint,render_template, request, redirect, \
	 url_for, jsonify
from flask_login import current_user, login_required, logout_user
from werkzeug.utils import secure_filename
from werkzeug.datastructures import CombinedMultiDict
from app.models import User, Dataset
from app.utils.SQL import SQLHelper
from app import db, celery
from app.main.forms import UploadForms, SelectDSForm, SelectDSFormSSA
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

result_nefs = None
result_ssa = None


@celery.task(bind=True)
def backgroudNE_task(self, X, Y, k, w, L, n_bands):
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

	# int_list = [int(x) for x in S]
	return {'current': len(S), 'total': n_bands, 'status': '任务完成', 'result': str(S)}


@celery.task(bind=True)
def backgroudSSA_task(self, dataset, labels, population, max_iter, times):
	fitness_list = list()
	index_list = list()
	for i in range(times):
		selected, best_fitness = QSSA(n_population=population, max_iter=max_iter, dataset=dataset, labels=labels, trans_type=7, ub=1, lb=0)
		index_list.append(selected)
		fitness_list.append(best_fitness)
		message = 'progressing'
		self.update_state(state='PROGRESS',
                          meta={'current': i, 'total': times, 'status': message})
		time.sleep(1)
	avg = np.mean(fitness_list)
	std = np.std(fitness_list, ddof=1)
	worst = max(fitness_list)
	best = min(fitness_list)
	best_index = index_list[fitness_list.index(best)]
	results = 'Avg:{} Std:{} Best:{} Worst:{} \nSelected: {}'.format(avg, std, best, worst, best_index)
	return {'current': 100, 'total': 100, 'status': 'Task completed!', 'result': results}


@hm.before_request
def before_request():
	if current_user.is_authenticated:
		current_user.last_seen = datetime.now()
		db.session.commit()


@hm.route('/index', methods=['GET', 'POST'])
@login_required
def index():
	return render_template('index.html')


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
			db_path, lb_path, k, w, L, n_bands = get_algorithm_args(form, type='ne')
			X, y = get_data(db_path[0], lb_path[0])
			global result_nefs
			result_nefs = [X, y, k, w, L, n_bands]
			return '成功提交，可以执行算法'
	return render_template('nebs.html', form=form)


@hm.route('/NEFS_task', methods=['POST'])
@login_required
def nefstask():
	global result_nefs
	task = backgroudNE_task.apply_async(
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
	task = backgroudNE_task.AsyncResult(task_id)
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
			db_path, lb_path, pop, max_iter, times = get_algorithm_args(form, type='ssa')
			X, y = get_data(db_path[0], lb_path[0])
			result_ssa = [X, y, pop, max_iter, times]
			return '成功提交，可以执行算法'
	return render_template('qssa.html', form=form)


@hm.route('/QSSA_task', methods=['POST'])
def qssatask():
	global result_ssa
	task = backgroudSSA_task.apply_async(
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
	task = backgroudSSA_task.AsyncResult(task_id)
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


