import subprocess
from . import app

from flask import jsonify


@app.route("/")
def hello():
    return jsonify({"message": "Hello, MRC topper!"})


@app.route("/prediction")
def prediction():
    subprocess.call(["bash", "./prediction.sh"])
    return jsonify({"message": "noice prediction"})
