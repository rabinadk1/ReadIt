import subprocess
from . import app
import json
from flask import jsonify, request


@app.route("/")
def hello():
    return jsonify({"message": "Hello, MRC topper!"})


@app.route("/prediction", methods=("GET", "POST"))
def prediction():
    predict_data = json.dumps(request.get_json())
    subprocess.run(
        [
            f"python run_squad.py --predict_data '{predict_data}'"
        ],
        shell=True,
    )
    return jsonify({"message": "noice prediction"})