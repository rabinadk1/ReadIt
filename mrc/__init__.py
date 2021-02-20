import os
import subprocess
import json

from flask import Flask
from run_squad import load_model

app = Flask(__name__, instance_relative_config=True)

app.config.from_pyfile("config.py", silent=True)

# ensure the instance folder exists
try:
    os.makedirs(app.instance_path)
except OSError:
    pass

with open(os.path.join(__name__, "model_config.json")) as config_file:
    app.config["model_config"] = json.load(config_file)

args, model, tokenizer, global_step = load_model(app.config["model_config"])
from . import routes
