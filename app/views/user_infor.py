#!/user/bin/env/ python
# -*- coding:utf-8 -*-

# @Time   : 2021/7/19 15:45
# @Author : Dong
# @File   : user_infor.py

from flask import Blueprint,render_template, request, session, redirect, \
	g, url_for
from flask_login import current_user, login_required, logout_user
from app.models import User

ui = Blueprint('user_infor', __name__)

