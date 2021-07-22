#!/user/bin/env/ python
# -*- coding:utf-8 -*-

# @Time   : 2021/7/19 16:18
# @Author : Dong
# @File   : __init__.py

from flask import Blueprint

bp = Blueprint('errors', __name__)

from app.errors import handlers