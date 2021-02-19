import subprocess
from . import app

from flask import jsonify

# a simple page that says hello
@app.route("/")
def hello():
    return jsonify({"message": "Hello, MRC topper!"})


@app.route("/prediction", methods=["GET", "POST"])
def prediction():
    subprocess.call(["bash", "./prediction.sh"])
    return jsonify({"message": "noice prediction"})
