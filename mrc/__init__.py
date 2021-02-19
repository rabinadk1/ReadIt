import os
import subprocess

from flask import Flask
from run_squad import main

app = Flask(__name__, instance_relative_config=True)

app.config.from_pyfile("config.py", silent=True)

# ensure the instance folder exists
try:
    os.makedirs(app.instance_path)
except OSError:
    pass

global args, model, tokenizer
# args, model, tokenizer = subprocess.check_output(
#     "python ./run_squad.py --model_type albert \
#     --model_name_or_path ./model4 --do_eval \
#         --do_lower_case --version_2_with_negative\
#               --max_answer_length=30 --doc_stride 128 \
#                   --max_query_length=64 --per_gpu_eval_batch_size=16 \
#                       --output_dir ../prediction4"
# )

from . import routes
