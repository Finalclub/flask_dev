#!/user/bin/env/ python
# -*- coding:utf-8 -*-

# @Time   : 2021/7/11 15:50
# @Author : Dong
# @File   : __init__.py

from flask import Flask
from flask_session import Session
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager

db = SQLAlchemy()
login = LoginManager()


def create_app():
	app = Flask(__name__)
	app.config.from_object('settings.DevelopmentConfig')

	db.init_app(app)
	login.init_app(app)
	login.login_view = 'login'

	from app.auth import account
	app.register_blueprint(account.ac)
	from app.views import home
	app.register_blueprint(home.hm)

	Session(app=app)

	return app


from app import models


if __name__ == '__main__':
	print(create_app())