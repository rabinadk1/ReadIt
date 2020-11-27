"""Official evaluation script for SQuAD version 2.0.

In addition to basic functionality, we also compute additional statistics and
plot precision-recall curves if an additional na_prob.json file is provided.
This file is expected to map question ID's to the model's predicted probability
that a question is unanswerable.
"""
import argparse
import collections
import json
import os
import re
import string
import sys
from typing import List, Mapping, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        "Official evaluation script for SQuAD version 2.0."
    )
    parser.add_argument(
        "data_file", metavar="dev-v2.0.json", help="Input data JSON file."
    )
    parser.add_argument(
        "pred_file", metavar="predictions.json", help="Model predictions."
    )
    parser.add_argument(
        "--out-file",
        "-o",
        metavar="eval.json",
        help="Write accuracy metrics to file (default is stdout).",
    )
    parser.add_argument(
        "--na-prob-file",
        "-n",
        metavar="na_prob.json",
        help="Model estimates of probability of no answer.",
    )
    parser.add_argument(
        "--na-prob-thresh",
        "-t",
        type=float,
        default=0.0,
        help='Predict "" if no-answer probability exceeds this (default = 1.0).',
    )
    parser.add_argument(
        "--out-image-dir",
        "-p",
        metavar="out_images",
        default=None,
        help="Save precision-recall curves to directory.",
    )
    parser.add_argument("--verbose", "-v", action="store_true")
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    return parser.parse_args()


def make_qid_to_has_ans(dataset):
    qid_to_has_ans = {}
    for article in dataset:
        for p in article["paragraphs"]:
            for qa in p["qas"]:
                qid_to_has_ans[qa["id"]] = bool(qa["answers"])
    return qid_to_has_ans


def normalize_answer(s: str) -> str:
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        regex = re.compile(r"\b(a|an|the)\b", re.UNICODE)
        return re.sub(regex, " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def get_tokens(s: str) -> Union[str, list]:
    if not s:
        return []
    return normalize_answer(s).split()


def compute_exact(a_gold: str, a_pred: str) -> int:
    return int(normalize_answer(a_gold) == normalize_answer(a_pred))


def compute_f1(a_gold: str, a_pred: str) -> Union[float, int]:
    gold_toks = get_tokens(a_gold)
    pred_toks = get_tokens(a_pred)
    common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
    num_same = sum(common.values())
    if len(gold_toks) == 0 or len(pred_toks) == 0:
        # If either is no-answer, then F1 is 1 if they agree, 0 otherwise
        return int(gold_toks == pred_toks)
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(pred_toks)
    recall = 1.0 * num_same / len(gold_toks)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def get_raw_scores(dataset, preds) -> Tuple[dict, dict]:
    exact_scores = {}
    f1_scores = {}
    for article in dataset:
        for p in article["paragraphs"]:
            for qa in p["qas"]:

                qid = qa["id"]
                if qid not in preds:
                    print(f"Missing prediction for {qid}")
                    continue

                gold_answers = [
                    a["text"] for a in qa["answers"] if normalize_answer(a["text"])
                ]
                if not gold_answers:
                    # For unanswerable questions, only correct answer is empty string
                    gold_answers = [""]
                a_pred = preds[qid]
                # Take max over all gold answers
                exact_scores[qid] = max(compute_exact(a, a_pred) for a in gold_answers)
                f1_scores[qid] = max(compute_f1(a, a_pred) for a in gold_answers)
    return exact_scores, f1_scores


def apply_no_ans_threshold(
    scores: dict, na_probs: dict, qid_to_has_ans: dict, na_prob_thresh: float
) -> dict:
    new_scores = {}
    for qid, s in scores.items():
        new_scores[qid] = (
            float(not qid_to_has_ans[qid]) if na_probs[qid] > na_prob_thresh else s
        )
    return new_scores


def make_eval_dict(
    exact_scores: dict, f1_scores: dict, qid_list: Optional[list] = None
) -> collections.OrderedDict:
    if not qid_list:
        total = len(exact_scores)
        return collections.OrderedDict(
            [
                ("exact", 100.0 * sum(exact_scores.values()) / total),
                ("f1", 100.0 * sum(f1_scores.values()) / total),
                ("total", total),
            ]
        )
    else:
        total = len(qid_list)
        return collections.OrderedDict(
            [
                ("exact", 100.0 * sum(exact_scores[k] for k in qid_list) / total),
                ("f1", 100.0 * sum(f1_scores[k] for k in qid_list) / total),
                ("total", total),
            ]
        )


def merge_eval(main_eval: dict, new_eval: dict, prefix: str):
    for k in new_eval:
        main_eval[f"{prefix}_{k}"] = new_eval[k]


def plot_pr_curve(
    precisions: List[float], recalls: List[float], out_image: str, title: str
):
    plt.step(recalls, precisions, color="b", alpha=0.2, where="post")
    plt.fill_between(recalls, precisions, step="post", alpha=0.2, color="b")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.xlim([0.0, 1.05])
    plt.ylim([0.0, 1.05])
    plt.title(title)
    plt.savefig(out_image)
    plt.clf()


def make_precision_recall_eval(
    scores: dict,
    na_probs: dict,
    num_true_pos: int,
    qid_to_has_ans: dict,
    out_image: Optional[str] = None,
    title: Optional[str] = None,
) -> Mapping[str, float]:
    qid_list = sorted(na_probs, key=lambda k: na_probs[k])
    true_pos = 0.0
    cur_p = 1.0
    cur_r = 0.0
    precisions = [1.0]
    recalls = [0.0]
    avg_prec = 0.0
    for i, qid in enumerate(qid_list):
        if qid_to_has_ans[qid]:
            true_pos += scores[qid]
        cur_p = true_pos / (i + 1)
        cur_r = true_pos / num_true_pos
        if i == len(qid_list) - 1 or na_probs[qid] != na_probs[qid_list[i + 1]]:
            # i.e., if we can put a threshold after this point
            avg_prec += cur_p * (cur_r - recalls[-1])
            precisions.append(cur_p)
            recalls.append(cur_r)
    if out_image:
        plot_pr_curve(precisions, recalls, out_image, title)
    return {"ap": 100 * avg_prec}


def run_precision_recall_analysis(
    main_eval: dict,
    exact_raw: dict,
    f1_raw: dict,
    na_probs: dict,
    qid_to_has_ans: dict,
    out_image_dir: str,
):
    if not os.path.exists(out_image_dir):
        os.makedirs(out_image_dir)
    num_true_pos = sum(1 for v in qid_to_has_ans.values() if v)
    if num_true_pos == 0:
        return
    pr_exact = make_precision_recall_eval(
        exact_raw,
        na_probs,
        num_true_pos,
        qid_to_has_ans,
        out_image=os.path.join(out_image_dir, "pr_exact.png"),
        title="Precision-Recall curve for Exact Match score",
    )
    pr_f1 = make_precision_recall_eval(
        f1_raw,
        na_probs,
        num_true_pos,
        qid_to_has_ans,
        out_image=os.path.join(out_image_dir, "pr_f1.png"),
        title="Precision-Recall curve for F1 score",
    )
    oracle_scores = {k: float(v) for k, v in qid_to_has_ans.items()}
    pr_oracle = make_precision_recall_eval(
        oracle_scores,
        na_probs,
        num_true_pos,
        qid_to_has_ans,
        out_image=os.path.join(out_image_dir, "pr_oracle.png"),
        title="Oracle Precision-Recall curve (binary task of HasAns vs. NoAns)",
    )
    merge_eval(main_eval, pr_exact, "pr_exact")
    merge_eval(main_eval, pr_f1, "pr_f1")
    merge_eval(main_eval, pr_oracle, "pr_oracle")


def histogram_na_prob(na_probs: dict, qid_list: list, image_dir: str, name: str):
    if not qid_list:
        return
    x = [na_probs[k] for k in qid_list]
    weights = np.ones_like(x) / float(len(x))
    plt.hist(x, weights=weights, bins=20, range=(0.0, 1.0))
    plt.xlabel("Model probability of no-answer")
    plt.ylabel("Proportion of dataset")
    plt.title(f"Histogram of no-answer probability: {name}")
    plt.savefig(os.path.join(image_dir, f"na_prob_hist_{name}.png"))
    plt.clf()


def find_best_thresh(
    preds: Union[list, dict], scores: dict, na_probs: dict, qid_to_has_ans: dict
) -> Tuple[float, float]:
    num_no_ans = sum(1 for k in qid_to_has_ans if not qid_to_has_ans[k])
    cur_score = num_no_ans
    best_score = cur_score
    best_thresh = 0.0
    qid_list = sorted(na_probs, key=lambda k: na_probs[k])
    for qid in qid_list:
        if qid not in scores:
            continue
        if qid_to_has_ans[qid]:
            diff = scores[qid]
        elif preds[qid]:
            diff = -1
        else:
            diff = 0
        cur_score += diff
        if cur_score > best_score:
            best_score = cur_score
            best_thresh = na_probs[qid]
    return 100 * best_score / len(scores), best_thresh


def find_all_best_thresh(
    main_eval: dict,
    preds: Union[dict, list],
    exact_raw: dict,
    f1_raw: dict,
    na_probs: dict,
    qid_to_has_ans: dict,
):
    best_exact, exact_thresh = find_best_thresh(
        preds, exact_raw, na_probs, qid_to_has_ans
    )
    best_f1, f1_thresh = find_best_thresh(preds, f1_raw, na_probs, qid_to_has_ans)
    main_eval["best_exact"] = best_exact
    main_eval["best_exact_thresh"] = exact_thresh
    main_eval["best_f1"] = best_f1
    main_eval["best_f1_thresh"] = f1_thresh


def eval_squad(
    data_file: str,
    pred_file: str,
    na_prob_file: Optional[str],
    na_prob_thresh: float,
    out_image_dir: Optional[str] = None,
):
    with open(data_file) as f:
        dataset_json: dict = json.load(f)
        dataset: Union[dict, list] = dataset_json["data"]
    with open(pred_file) as f:
        preds: Union[dict, list] = json.load(f)
    if na_prob_file:
        with open(na_prob_file) as f:
            na_probs: dict = json.load(f)
    else:
        na_probs = {k: 0.0 for k in preds}
    qid_to_has_ans = make_qid_to_has_ans(dataset)  # maps qid to True/False
    has_ans_qids = [k for k, v in qid_to_has_ans.items() if v]
    no_ans_qids = [k for k, v in qid_to_has_ans.items() if not v]
    exact_raw, f1_raw = get_raw_scores(dataset, preds)
    exact_thresh = apply_no_ans_threshold(
        exact_raw, na_probs, qid_to_has_ans, na_prob_thresh
    )
    f1_thresh = apply_no_ans_threshold(f1_raw, na_probs, qid_to_has_ans, na_prob_thresh)
    out_eval = make_eval_dict(exact_thresh, f1_thresh)
    if has_ans_qids:
        has_ans_eval = make_eval_dict(exact_thresh, f1_thresh, qid_list=has_ans_qids)
        merge_eval(out_eval, has_ans_eval, "HasAns")
    if no_ans_qids:
        no_ans_eval = make_eval_dict(exact_thresh, f1_thresh, qid_list=no_ans_qids)
        merge_eval(out_eval, no_ans_eval, "NoAns")
    if na_prob_file:
        find_all_best_thresh(
            out_eval, preds, exact_raw, f1_raw, na_probs, qid_to_has_ans
        )
        if out_image_dir:
            run_precision_recall_analysis(
                out_eval, exact_raw, f1_raw, na_probs, qid_to_has_ans, out_image_dir
            )
            histogram_na_prob(na_probs, has_ans_qids, out_image_dir, "hasAns")
            histogram_na_prob(na_probs, no_ans_qids, out_image_dir, "noAns")

    return out_eval
