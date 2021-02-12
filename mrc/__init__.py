import os

from flask import Flask


app = Flask(__name__, instance_relative_config=True)

app.config.from_pyfile("config.py", silent=True)

# ensure the instance folder exists
try:
    os.makedirs(app.instance_path)
except OSError:
    pass

from . import routes
