#!/bin/bash
python3 run_squad.py \
    --model_type albert \
    --model_name_or_path ./model4 \
    --do_eval \
    --do_lower_case \
    --version_2_with_negative \
    --predict_file ./dev-v2.0modified.json \
    --max_seq_length 512 \
    --n_best_size=20 \
    --max_answer_length=30 \
    --doc_stride 128 \
    --max_query_length=64 \
    --per_gpu_eval_batch_size=16 \
    --output_dir ../prediction4