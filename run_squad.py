# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Finetuning the library models for question-answering on SQuAD
(DistilBERT, Bert, XLM, XLNet).
"""

from __future__ import absolute_import, division, print_function

import glob
import logging
import os
import timeit
import collections

import torch
from torch.utils.data import DataLoader, SequentialSampler
from tqdm import tqdm
from transformers import (
    WEIGHTS_NAME,
    AlbertConfig,
    AlbertForQuestionAnswering,
    AlbertTokenizer,
    BertConfig,
    BertForQuestionAnswering,
    BertTokenizer,
    DistilBertConfig,
    DistilBertForQuestionAnswering,
    DistilBertTokenizer,
    XLMConfig,
    XLMForQuestionAnswering,
    XLMTokenizer,
    XLNetConfig,
    XLNetForQuestionAnswering,
    XLNetTokenizer,
    squad_convert_examples_to_features,
)
from .custom_predict import (
    custom_compute_predictions_log_probs,
    custom_compute_predictions_logits,
)
from transformers.data.processors.squad import SquadResult, SquadV1Processor

# from evaluate_official2 import eval_squad
from run_base import BaseParser, CustomSquadV2Processor, base_main, base_train

logger = logging.getLogger(__name__)

ALL_MODELS = sum(
    (
        tuple(conf.pretrained_config_archive_map.keys())
        for conf in (BertConfig, XLNetConfig, XLMConfig)
    ),
    (),
)

MODEL_CLASSES = {
    "bert": (BertConfig, BertForQuestionAnswering, BertTokenizer),
    "xlnet": (XLNetConfig, XLNetForQuestionAnswering, XLNetTokenizer),
    "xlm": (XLMConfig, XLMForQuestionAnswering, XLMTokenizer),
    "distilbert": (
        DistilBertConfig,
        DistilBertForQuestionAnswering,
        DistilBertTokenizer,
    ),
    "albert": (AlbertConfig, AlbertForQuestionAnswering, AlbertTokenizer),
}


def to_list(tensor):
    return tensor.detach().cpu().tolist()


def train(args, train_dataset, model, tokenizer):
    """ Train the model """
    (
        train_dataloader,
        amp,
        optimizer,
        scheduler,
        tb_writer,
        tr_loss,
        logging_loss,
        train_iterator,
    ) = base_train(args, train_dataset, model, logger)

    global_step = 1

    for _ in train_iterator:
        epoch_iterator = tqdm(
            train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0]
        )
        for step, batch in enumerate(epoch_iterator):
            model.train()
            batch = tuple(t.to(args.device) for t in batch)

            inputs = {
                "input_ids": batch[0],
                "attention_mask": batch[1],
                "start_positions": batch[3],
                "end_positions": batch[4],
            }

            if args.model_type != "distilbert":
                inputs["token_type_ids"] = (
                    None if args.model_type == "xlm" else batch[2]
                )

            if args.model_type in ["xlnet", "xlm"]:
                inputs.update({"cls_index": batch[6], "p_mask": batch[7]})

            outputs = model(**inputs)
            loss = outputs[
                0
            ]  # model outputs are always tuple in transformers (see doc)

            if args.n_gpu > 1:
                # mean() to average on multi-gpu parallel (not distributed) training
                loss = loss.mean()
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            tr_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(
                    amp.master_params(optimizer) if args.fp16 else model.parameters(),
                    args.max_grad_norm,
                )

                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1

                if args.local_rank in [-1, 0]:
                    # Log metrics
                    if args.logging_steps > 0 and global_step % args.logging_steps == 0:
                        if args.local_rank == -1 and args.evaluate_during_training:
                            # Only evaluate when single GPU
                            # otherwise metrics may not average well
                            results = evaluate(args, model, tokenizer)
                            for key, value in results.items():
                                tb_writer.add_scalar(f"eval_{key}", value, global_step)
                        tb_writer.add_scalar("lr", scheduler.get_lr()[0], global_step)
                        tb_writer.add_scalar(
                            "loss",
                            (tr_loss - logging_loss) / args.logging_steps,
                            global_step,
                        )
                        logging_loss = tr_loss

                    # Save model checkpoint
                    if args.save_steps > 0 and global_step % args.save_steps == 0:
                        output_dir = os.path.join(
                            args.output_dir, f"checkpoint-{global_step}"
                        )
                        if not os.path.exists(output_dir):
                            os.makedirs(output_dir)
                        model_to_save = (
                            model.module if hasattr(model, "module") else model
                        )  # Take care of distributed/parallel training
                        model_to_save.save_pretrained(output_dir)
                        torch.save(args, os.path.join(output_dir, "training_args.bin"))
                        logger.info(f"Saving model checkpoint to {output_dir}")

            if args.max_steps > 0 and global_step > args.max_steps:
                epoch_iterator.close()
                break
        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break

    if args.local_rank in [-1, 0]:
        tb_writer.close()

    return global_step, tr_loss / global_step


def evaluate(args, model, tokenizer, prefix=""):
    dataset, examples, features = load_and_cache_examples(
        args, tokenizer, evaluate=True, output_examples=True
    )

    # NOTE: added to compute the prediction
    prediction = collections.OrderedDict()

    if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
        os.makedirs(args.output_dir)

    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)

    # Note that DistributedSampler samples randomly
    eval_sampler = SequentialSampler(dataset)
    eval_dataloader = DataLoader(
        dataset, sampler=eval_sampler, batch_size=args.eval_batch_size
    )

    # multi-gpu evaluate
    if args.n_gpu > 1 and not isinstance(model, torch.nn.DataParallel):
        model = torch.nn.DataParallel(model)

    # Eval!
    logger.info(f"***** Running evaluation {prefix} *****")
    logger.info(f"  Num examples = {len(dataset)}")
    logger.info(f"  Batch size = {args.eval_batch_size}")

    all_results = []
    start_time = timeit.default_timer()

    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        model.eval()
        batch = tuple(t.to(args.device) for t in batch)

        with torch.no_grad():
            inputs = {"input_ids": batch[0], "attention_mask": batch[1]}

            if args.model_type != "distilbert":
                inputs["token_type_ids"] = (
                    None if args.model_type == "xlm" else batch[2]
                )  # XLM don't use segment_ids

            example_indices = batch[3]

            # XLNet and XLM use more arguments for their predictions
            if args.model_type in ["xlnet", "xlm"]:
                inputs.update({"cls_index": batch[5], "p_mask": batch[6]})

            outputs = model(**inputs)

        for i, example_index in enumerate(example_indices):
            eval_feature = features[example_index.item()]
            unique_id = int(eval_feature.unique_id)

            output = [to_list(output[i]) for output in outputs]

            # Some models (XLNet, XLM) use 5 arguments for their predictions,
            # while the other "simpler" models only use two.
            if len(output) >= 5:
                (
                    start_logits,
                    start_top_index,
                    end_logits,
                    end_top_index,
                    cls_logits,
                ) = output[:5]

                result = SquadResult(
                    unique_id,
                    start_logits,
                    end_logits,
                    start_top_index=start_top_index,
                    end_top_index=end_top_index,
                    cls_logits=cls_logits,
                )

            else:
                start_logits, end_logits = output
                result = SquadResult(unique_id, start_logits, end_logits)

            all_results.append(result)

    evalTime = timeit.default_timer() - start_time
    logger.info(
        f"  Evaluation done in total {evalTime} secs "
        f"({evalTime / len(dataset)} sec per example)",
    )

    # Compute predictions
    # output_prediction_file = os.path.join(args.output_dir, f"predictions_{prefix}.json")
    # output_nbest_file = os.path.join(
    #     args.output_dir, "nbest_predictions_{}.json".format(prefix)
    # )

    # output_null_log_odds_file = (
    #     os.path.join(args.output_dir, f"null_odds_{prefix}.json")
    #     if args.version_2_with_negative
    #     else None
    # )

    # XLNet and XLM use a more complex post-processing procedure
    if args.model_type in ["xlnet", "xlm"]:
        start_n_top = (
            model.config.start_n_top
            if hasattr(model, "config")
            else model.module.config.start_n_top
        )
        end_n_top = (
            model.config.end_n_top
            if hasattr(model, "config")
            else model.module.config.end_n_top
        )

        prediction = custom_compute_predictions_log_probs(
            examples,
            features,
            all_results,
            args.n_best_size,
            args.max_answer_length,
            start_n_top,
            end_n_top,
            args.version_2_with_negative,
            tokenizer,
            args.verbose_logging,
        )
    else:
        prediction = custom_compute_predictions_logits(
            examples,
            features,
            all_results,
            args.n_best_size,
            args.max_answer_length,
            args.do_lower_case,
            args.verbose_logging,
            args.version_2_with_negative,
            args.null_score_diff_threshold,
        )

    # Compute the F1 and exact scores.
    # results = squad_evaluate(examples, predictions)
    # SQuAD 2.0
    # results = eval_squad(
    #     os.path.join(args.data_dir, args.predict_file),
    #     output_prediction_file,
    #     output_null_log_odds_file,
    #     args.null_score_diff_threshold,
    # )
    # return results
    return prediction


def load_and_cache_examples(args, tokenizer, evaluate=False, output_examples=False):
    if args.local_rank not in [-1, 0] and not evaluate:
        # Make sure only the first process in distributed training
        # process the dataset, and the others will use the cache
        torch.distributed.barrier()

    # Load data features from cache or dataset file
    input_dir = args.data_dir if args.data_dir else "."

    cached_features_file = os.path.join(
        input_dir,
        f"cached_{'dev' if evaluate else 'train'}_"
        f"{list(filter(None, args.model_name_or_path.split('/'))).pop()}_"
        f"{args.max_seq_length}",
    )

    # Init features and dataset from cache if it exists
    if (
        os.path.exists(cached_features_file)
        and not args.overwrite_cache
        and not output_examples
    ):
        logger.info(f"Loading features from cached file {cached_features_file}")
        features_and_dataset = torch.load(cached_features_file)
        features, dataset = (
            features_and_dataset["features"],
            features_and_dataset["dataset"],
        )
    else:
        logger.info(f"Creating features from dataset file at {input_dir}")

        if not args.data_dir and (
            (evaluate and not args.predict_file)
            or (not evaluate and not args.train_file)
        ):
            try:
                import tensorflow_datasets as tfds
            except ImportError:
                raise ImportError(
                    "If not data_dir is specified, tensorflow_datasets "
                    "needs to be installed."
                )

            if args.version_2_with_negative:
                logger.warn("tensorflow_datasets does not handle version 2 of SQuAD.")

            tfds_examples = tfds.load("squad")
            examples = SquadV1Processor().get_examples_from_dataset(
                tfds_examples, evaluate=evaluate
            )
        else:
            processor = (
                CustomSquadV2Processor()
                if args.version_2_with_negative
                else SquadV1Processor()
            )

            examples = (
                processor.get_dev_examples(args.data_dir, filename=args.predict_file)
                if evaluate
                else processor.get_train_examples(
                    args.data_dir, filename=args.train_file
                )
            )

        features, dataset = squad_convert_examples_to_features(
            examples=examples,
            tokenizer=tokenizer,
            max_seq_length=args.max_seq_length,
            doc_stride=args.doc_stride,
            max_query_length=args.max_query_length,
            is_training=not evaluate,
            return_dataset="pt",
        )

        if args.local_rank in [-1, 0]:
            logger.info(f"Saving features into cached file {cached_features_file}")
            torch.save({"features": features, "dataset": dataset}, cached_features_file)

    if args.local_rank == 0 and not evaluate:
        # Make sure only the first process in distributed training
        # process the dataset, and the others will use the cache
        torch.distributed.barrier()

    if output_examples:
        return dataset, examples, features
    return dataset


def main():
    parser = BaseParser(ALL_MODELS, MODEL_CLASSES)

    # !Other parameters
    parser.add_argument(
        "--padding_side",
        default="right",
        type=str,
        help="right/left, padding_side of passage / question",
    )
    parser.add_argument(
        "--data_dir",
        default="",
        type=str,
        help=(
            "The input data dir. Should contain the .json files for the task."
            "If no data dir or train/predict files are specified, will run "
            "with tensorflow_datasets."
        ),
    )
    parser.add_argument(
        "--train_file",
        default=None,
        type=str,
        help=(
            "The input training file. If a data dir is specified, will look for "
            "the file there.If no data dir or train/predict files are specified, "
            "will run with tensorflow_datasets."
        ),
    )
    parser.add_argument(
        "--predict_file",
        default=None,
        type=str,
        help=(
            "The input evaluation file. If a data dir is specified, will look for "
            "the file there. If no data dir or train/predict files are specified, "
            "will run with tensorflow_datasets."
        ),
    )

    parser.add_argument(
        "--version_2_with_negative",
        action="store_true",
        help="If true, the SQuAD examples contain some that do not have an answer.",
    )
    parser.add_argument(
        "--null_score_diff_threshold",
        type=float,
        default=0.0,
        help=(
            "If null_score - best_non_null is greater than "
            "the threshold predict null."
        ),
    )

    parser.add_argument(
        "--max_seq_length",
        default=384,
        type=int,
        help=(
            "The maximum total input sequence length after WordPiece tokenization. "
            "Sequences longer than this will be truncated, and "
            "sequences shorter than this will be padded."
        ),
    )
    parser.add_argument(
        "--doc_stride",
        default=128,
        type=int,
        help=(
            "When splitting up a long document into chunks, "
            "how much stride to take between chunks."
        ),
    )
    parser.add_argument(
        "--max_query_length",
        default=64,
        type=int,
        help=(
            "The maximum number of tokens for the question. "
            "Questions longer than this will be truncated to this length."
        ),
    )

    parser.add_argument(
        "--n_best_size",
        default=20,
        type=int,
        help=(
            "The total number of n-best predictions to generate in the "
            "nbest_predictions.json output file."
        ),
    )
    parser.add_argument(
        "--max_answer_length",
        default=30,
        type=int,
        help=(
            "The maximum length of an answer that can be generated. "
            "This is needed because the start and end predictions are not "
            "conditioned on one another."
        ),
    )
    parser.add_argument(
        "--verbose_logging",
        action="store_true",
        help=(
            "If true, all of the warnings related to data processing will be printed. "
            "A number of warnings are expected for a normal SQuAD evaluation."
        ),
    )

    args, model, model_class, tokenizer, tokenizer_class = base_main(
        parser, logger, MODEL_CLASSES, is_cls=False
    )

    # Before we do anything with models, we want to ensure that we get fp16 execution
    # of torch.einsum if args.fp16 is set.
    # Otherwise it'll default to "promote" mode, and we'll get fp32 operations.
    # Note that running `--fp16_opt_level="O2"` will
    # remove the need for this code, but it is still valid.
    if args.fp16:
        try:
            import apex
        except ImportError:
            raise ImportError(
                (
                    "Please install apex from "
                    "https://www.github.com/nvidia/apex to use fp16 training."
                )
            )
        apex.amp.register_half_function(torch, "einsum")

    # Training
    if args.do_train:
        train_dataset = load_and_cache_examples(
            args, tokenizer, evaluate=False, output_examples=False
        )
        global_step, tr_loss = train(args, train_dataset, model, tokenizer)
        logger.info(f" global_step = {global_step}, average loss = {tr_loss}")

        # Save the trained model and the tokenizer
        if args.local_rank == -1 or torch.distributed.get_rank() == 0:
            # Create output directory if needed
            if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
                os.makedirs(args.output_dir)

            logger.info(
                f"Saving model checkpoint to {args.output_dir}",
            )
            # Save a trained model, configuration and
            # tokenizer using `save_pretrained()`.
            # They can then be reloaded using `from_pretrained()`
            model_to_save = (
                model.module if hasattr(model, "module") else model
            )  # Take care of distributed/parallel training
            model_to_save.save_pretrained(args.output_dir)
            tokenizer.save_pretrained(args.output_dir)

            # Good practice: save your training arguments
            # together with the trained model
            torch.save(args, os.path.join(args.output_dir, "training_args.bin"))

            # Load a trained model and vocabulary that you have fine-tuned
            model = model_class.from_pretrained(args.output_dir, force_download=True)
            tokenizer = tokenizer_class.from_pretrained(
                args.output_dir, do_lower_case=args.do_lower_case
            )
        model.to(args.device)

    # Evaluation - we can ask to evaluate all the checkpoints
    # (sub-directories) in a directory
    # results = {}
    if args.do_eval and args.local_rank in [-1, 0]:

        if args.do_train:
            logger.info("Loading checkpoints saved during training for evaluation")
            checkpoints = [args.output_dir]
            if args.eval_all_checkpoints:
                checkpoints = list(
                    os.path.dirname(c)
                    for c in sorted(
                        glob.glob(
                            args.output_dir + "/**/" + WEIGHTS_NAME, recursive=True
                        )
                    )
                )
                logging.getLogger("transformers.modeling_utils").setLevel(
                    logging.WARN
                )  # Reduce model loading logs
        elif args.eval_all_checkpoints:
            checkpoints = list(
                os.path.dirname(c)
                for c in sorted(
                    glob.glob(args.output_dir + "/**/" + WEIGHTS_NAME, recursive=True)
                )
            )
            logger.info(f"Loading checkpoint {checkpoints} for evaluation")
            logging.getLogger("transformers.modeling_utils").setLevel(
                logging.WARN
            )  # Reduce model loading logs
        else:
            logger.info(f"Loading checkpoint {args.model_name_or_path} for evaluation")
            checkpoints = [args.model_name_or_path]

        logger.info(f"Evaluate the following checkpoints: {checkpoints}")

        for checkpoint in checkpoints:
            # Reload the model
            global_step = checkpoint.split("-")[-1] if len(checkpoints) > 1 else ""
            model = model_class.from_pretrained(checkpoint, force_download=True)
            model.to(args.device)

            # Evaluate
            # result = evaluate(args, model, tokenizer, prefix=global_step)
            evaluate(args, model, tokenizer, prefix=global_step)

            # result = {
            #     (f"{k}_{global_step}" if global_step else k): v
            #     for k, v in result.items()
            # }
            # results.update(result)

    # logger.info(f"Results: {results}")
    # with open(os.path.join(args.output_dir, "result.txt"), "a") as writer:
    #     for key in sorted(results):
    #         logger.info(f"  {key} = {results[key]}")
    #         writer.write(f"{key} = {results[key]}\t")
    #         writer.write("\t\n")
    # return results


if __name__ == "__main__":
    main()
