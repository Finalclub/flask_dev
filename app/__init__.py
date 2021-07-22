#!/user/bin/env/ python
# -*- coding:utf-8 -*-

# @Time   : 2021/7/11 15:50
# @Author : Dong
# @File   : __init__.py

from flask import Flask
from flask_session import Session
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager
from flask_migrate import Migrate
# from flask_redis import FlaskRedis
from celery import Celery

db = SQLAlchemy()
login = LoginManager()
# redis = FlaskRedis()
celery = Celery(__name__, broker='redis://:luka@127.0.0.1:6379/1', result_backend='redis://:luka@127.0.0.1:6379/2')
# files = UploadSet('files', DOCUMENTS)


def make_celery(app):

	class ContextTask(celery.Task):
		def __call__(self, *args, **kwargs):
			with app.app_context():
				return self.run(*args, **kwargs)

	celery.Task = ContextTask


def create_app():
	app = Flask(__name__)
	app.config.from_object('settings.DevelopmentConfig')

	db.init_app(app)
	login.init_app(app)
	# redis.init_app(app)
	app.config.update(
		CELERY_BROKER_URL='redis://:luka@127.0.0.1:6379/1',
		CELERY_RESULT_BACKEND='redis://:luka@127.0.0.1:6379/2'
	)
	make_celery(app)
	# configure_uploads(app, files)
	login.login_view = 'login'
	# celery.__init__(app.name, broker=app.config['CELERY_BROKER_URL'])
	# celery.conf.update(app.config)

	from app.auth import account
	app.register_blueprint(account.ac)
	from app.views import home
	app.register_blueprint(home.hm)
	migrate = Migrate(app, db)

	Session(app=app)

	return app


from app import models