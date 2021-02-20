import subprocess
from . import app, args, model, tokenizer, global_step
import json
from flask import jsonify, request
from run_squad import evaluate


@app.route("/")
def hello():
    return jsonify({"message": "Hello, MRC topper!"})


@app.route("/prediction", methods=("GET", "POST"))
def prediction():
    predict_data = json.dumps(request.get_json())
    args.predict_data = predict_data
    prediction = evaluate(args, model, tokenizer, prefix=global_step)
    print(prediction)
    return jsonify({"message": "noice prediction", "prediction": prediction})
