#!/user/bin/env/ python
# -*- coding:utf-8 -*-

# @Time   : 2021/7/11 15:50
# @Author : Dong
# @File   : manage.py

from flask import Flask
from flask_session import Session
from views import account
from views import home
from exts.auth import Auth


def create_app():
	app = Flask(__name__)
	app.config.from_object('settings.DevelopmentConfig')


	app.register_blueprint(account.ac)
	app.register_blueprint(home.hm)

	# auth = Auth()
	# auth.init_app(app=app)
	Session(app=app)

	return app