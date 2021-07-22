在Windows平台Celery运行的时候报错：ValueError: not enough values to unpack

启动命令改为:
celery -A app.views.home.celery worker --pool=solo  --loglevel=info