import argparse
import logging
import os
import random
import json
from typing import Any, Mapping, Tuple

import numpy as np
import torch

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter

from torch.utils.data import DataLoader, RandomSampler
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange
from transformers import AdamW, get_linear_schedule_with_warmup
from transformers.data.processors.squad import SquadExample, SquadV2Processor


def set_seed(args: argparse.Namespace):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def base_train(
    args: argparse.Namespace, train_dataset, model, logger: logging.Logger
) -> Tuple[Any, Any, Any, Any, Any, float, float, Any]:
    """ Base for training the model """
    if args.local_rank in [-1, 0]:
        tb_writer = SummaryWriter()

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_sampler = (
        RandomSampler(train_dataset)
        if args.local_rank == -1
        else DistributedSampler(train_dataset)
    )
    train_dataloader = DataLoader(
        train_dataset, sampler=train_sampler, batch_size=args.train_batch_size
    )

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = (
            args.max_steps
            // (len(train_dataloader) // args.gradient_accumulation_steps)
            + 1
        )
    else:
        t_total = (
            len(train_dataloader)
            // args.gradient_accumulation_steps
            * args.num_train_epochs
        )

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]

    optimizer = AdamW(
        optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon
    )
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
    )

    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError(
                (
                    "Please install apex from https://www.github.com/nvidia/apex"
                    " to use fp16 training."
                )
            )

        model, optimizer = amp.initialize(
            model, optimizer, opt_level=args.fp16_opt_level
        )
    else:
        amp = None  # used here while transfering to base_train function

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            find_unused_parameters=True,
        )

    # Train!
    logger.info("***** Running training *****")
    logger.info(
        f"  Num examples = {len(train_dataset)}",
    )
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per GPU = {args.per_gpu_train_batch_size}")
    total_train_batch_size = (
        args.train_batch_size
        * args.gradient_accumulation_steps
        * (torch.distributed.get_world_size() if args.local_rank != -1 else 1)
    )
    logger.info(
        "  Total train batch size (w. parallel, distributed & accumulation) = "
        f"{total_train_batch_size}",
    )
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {t_total}")

    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    train_iterator = trange(
        int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0]
    )
    set_seed(args)  # Added here for reproductibility (even between python 2 and 3)

    return (
        train_dataloader,
        amp,
        optimizer,
        scheduler,
        tb_writer,
        tr_loss,
        logging_loss,
        train_iterator,
    )


def base_main(
    args: argparse.Namespace,
    logger: logging.Logger,
    MODEL_CLASSES: Mapping[str, tuple],
    is_cls: bool,
) -> Tuple[argparse.Namespace, Any, Any, Any, Any]:
    if is_cls:
        from transformers import glue_output_modes as output_modes
        from transformers import glue_processors as processors

    if (
        os.path.exists(args.output_dir)
        and os.listdir(args.output_dir)
        and args.do_train
        and not args.overwrite_output_dir
    ):
        raise ValueError(
            (
                f"Output directory ({args.output_dir}) already exists and is not empty."
                " Use --overwrite_output_dir to overcome."
            )
        )

    # Setup distant debugging if needed
    if args.server_ip and args.server_port:
        # Distant debugging - see
        # https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd

        print("Waiting for debugger attach")
        ptvsd.enable_attach(
            address=(args.server_ip, args.server_port), redirect_output=True
        )
        ptvsd.wait_for_attach()

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device(
            "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"
        )
        args.n_gpu = torch.cuda.device_count()
    else:
        # Initializes the distributed backend
        # which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl")
        args.n_gpu = 1
    args.device = device

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        (
            f"Process rank: {args.local_rank}, device: {device}, "
            f"n_gpu: {args.n_gpu}, distributed training: {args.local_rank != -1}, "
            f"16-bits training: {args.fp16}"
        )
    )

    # Set seed
    set_seed(args)

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        # Make sure only the first process in
        # distributed training will download model & vocab
        torch.distributed.barrier()

    args.model_type = args.model_type.lower()
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]

    cache_dir = args.cache_dir if args.cache_dir else None

    if is_cls:
        # Prepare GLUE task
        args.task_name = args.task_name.lower()
        if args.task_name not in processors:
            raise ValueError(f"Task not found: {args.task_name}")
        processor = processors[args.task_name]()
        args.output_mode = output_modes[args.task_name]
        label_list = processor.get_labels()
        num_labels = len(label_list)

        config = config_class.from_pretrained(
            args.config_name if args.config_name else args.model_name_or_path,
            num_labels=num_labels,
            finetuning_task=args.task_name,
            cache_dir=cache_dir,
        )
    else:
        config = config_class.from_pretrained(
            args.config_name if args.config_name else args.model_name_or_path,
            cache_dir=cache_dir,
        )

    tokenizer = tokenizer_class.from_pretrained(
        args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
        do_lower_case=args.do_lower_case,
        cache_dir=cache_dir,
    )

    model = model_class.from_pretrained(
        args.model_name_or_path,
        from_tf=".ckpt" in args.model_name_or_path,
        config=config,
        cache_dir=cache_dir,
    )

    if args.local_rank == 0:
        # Make sure only the first process in
        # distributed training will download model & vocab
        torch.distributed.barrier()

    model.to(args.device)

    logger.info(f"Training/evaluation parameters {args}")

    return args, model, model_class, tokenizer, tokenizer_class


# Made a custom processor to remove the answer loading part
# This helps in prediction
class CustomSquadV2Processor(SquadV2Processor):
    def _create_examples(self, input_data, set_type):
        is_training = set_type == "train"
        examples = []
        for entry in tqdm(input_data):
            title = entry["title"]
            for paragraph in entry["paragraphs"]:
                context_text = paragraph["context"]
                for qa in paragraph["qas"]:
                    qas_id = qa["id"]
                    question_text = qa["question"]
                    start_position_character = None
                    answer_text = None
                    answers = []

                    is_impossible = qa.get("is_impossible", False)

                    if not is_impossible:
                        if is_training:
                            answer = qa["answers"][0]
                            answer_text = answer["text"]
                            start_position_character = answer["answer_start"]
                        # else:
                        # answers = qa["answers"]

                    example = SquadExample(
                        qas_id=qas_id,
                        question_text=question_text,
                        context_text=context_text,
                        answer_text=answer_text,
                        start_position_character=start_position_character,
                        title=title,
                        is_impossible=is_impossible,
                        answers=answers,
                    )

                    examples.append(example)
        return examples

    def get_dev_examples(self, data_dir, predict_data=None):
        """
        Returns the evaluation example from the data directory.

        Args:
            data_dir: Directory containing the data files used for training and evaluating.
            filename: None by default, specify this if the evaluation file has a different name than the original one
                which is `train-v1.1.json` and `train-v2.0.json` for squad versions 1.1 and 2.0 respectively.
        """
        if data_dir is None:
            data_dir = ""

        if self.dev_file is None:
            raise ValueError(
                "SquadProcessor should be instantiated via SquadV1Processor or SquadV2Processor"
            )

        if predict_data is None:
            with open(
                os.path.join(data_dir, self.dev_file), "r", encoding="utf-8"
            ) as reader:
                input_data = json.load(reader)["data"]
        else:
            print(predict_data)
            input_data = json.loads(predict_data)["data"]
        return self._create_examples(input_data, "dev")
