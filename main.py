import os
import json
import copy
import time
import random

import torch
import deepspeed
import torch.distributed as dist
import torch.nn.functional as F

from tqdm import tqdm
from functools import partial
from typing import List, Dict, Tuple, Optional, Union, Callable, Iterable, Any, Callable
from datetime import timedelta

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, PreTrainedModel, GenerationConfig, PreTrainedTokenizer
from deepspeed.ops.adam import DeepSpeedCPUAdam
from torch.utils.data import Dataset

from openreviewer.arguments import get_args
from openreviewer.dataset import InstructionTuningDataset
from openreviewer.utils import vicuna_sample_processor, print_rank, broadcast_model, move_dict_to_device, save_checkpoint
from openreviewer.scheduler import CosineWarmUpScheduler


def get_dataset(args, tokenizer, process_func=vicuna_sample_processor) -> Dataset:
    if args.dataset_type == "InstructionTuningDataset":
        DatasetClass = InstructionTuningDataset
    else:
        raise NotImplementedError(f"Not implemented dataset: {args.dataset_type}.")
    dataset = DatasetClass(
        args,
        path=args.data_path,
        tokenizer=tokenizer,
        process_func=process_func
    )


def train(args, model, optimizer, scheduler, tokenizer, dataset):
    print("Start training")

    dp_world_size = dist.get_world_size()
    dp_rank = dist.get_rank()
    dp_group = None

    sampler = DistributedSampler(
        dataset,
        shuffle=True,
        drop_last=True,
        rank=dp_rank,
        num_replicas=dp_world_size
    )

    dataloader = DataLoader(
        dataset,
        sampler=sampler,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        collate_fn=dataset.collate
    )

    epoch = 0  # TODO

    model.train()
    model.gradient_checkpointing_enable()

    for iteration, (input_batch, gen_batch, other_batch) in enumerate(dataloader):
        global_loss = 0
        move_dict_to_device(input_batch, model.device)
        move_dict_to_device(other_batch, model.device)
        move_dict_to_device(gen_batch, model.device)

        loss_mask = other_batch['loss_mask']
        labels = other_batch['labels']
        logits = model(**input_batch, return_dict=True, use_cache=False).logits

        # compute loss
        losses = F.cross_entropy(logits.reshape(-1, logits.size(-1)), labels.reshape(-1, ), reduction='none').reshape(labels.shape)
        loss = (loss_mask * losses).sum() / loss_mask.sum()

        model.backward(loss)
        model.step()

        dist.all_reduce(loss, dist.ReduceOp.SUM, group=dp_group)
        global_loss += loss.item() / dp_world_size / args.gradient_accumulation_steps

        # logging
        print_rank(f"iteration: {iteration}, loss: {global_loss}")

        # cleaning
        move_dict_to_device(input_batch, torch.device('cpu'))
        move_dict_to_device(other_batch, torch.device('cpu'))
        move_dict_to_device(gen_batch, torch.device('cpu'))

    model.eval()
    save_checkpoint(save_path, model, tokenizer)


def main():
    args = get_args()
    args.rank = int(os.getenv("RANK", "0"))
    args.world_size = int(os.getenv("WORLD_SIZE", "1"))
    args.local_rank = int(os.getenv("LOCAL_RANK", "0"))
    torch.cuda.set_device(args.local_rank)
    
    print(f"Using world size: {args.world_size}")
    deepspeed.init_distributed(timeout=timedelta(minutes=300))

    if dist.get_rank() == 0:
        os.makedirs(args.save_path, exist_ok=True)

    seed = dist.get_rank() + args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.enabled = False

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)

    dataset = get_dataset(args, tokenizer)

    # load ds_config
    with open(args.deepspeed_config, 'r', encoding='utf-8') as f:
        ds_config = json.load(f)
        ds_config["train_micro_batch_size_per_gpu"] = args.batch_size
        ds_config["gradient_clipping"] = args.clip_grad
        ds_config["gradient_accumulation_steps"] = args.gradient_accumulation_steps

    # load model
    model = AutoModel.from_pretrained(args.model_path, trust_remote_code=True)
    model.cuda()

    optimizer = DeepSpeedCPUAdam(student_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    lr_scheduler = CosineWarmUpScheduler(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        total_steps=args.total_iters,
        eta_min=args.min_lr
    )

    model, optimizer, _, lr_scheduler = deepspeed.initialize(
        model=student_model,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        config=ds_config
    )

    training(
        args,
        model=model,
        optimizer=optimizer,
        scheduler=lr_scheduler,
        tokenizer=tokenizer,
        dataset=dataset
    )


if __name__ == "__main__":
    main()
