import json
import os

import torch

import torch.distributed as dist

from typing import List, Tuple, Dict, Union, Optional
from functools import partial

from transformers import PreTrainedTokenizer, PreTrainedModel, GenerationConfig

from openreviewer.common import vicuna_system_prompt


def build_vicuna_input(messages: List[Tuple], system_message: str=vicuna_system_prompt):
    # roles=("USER", "ASSISTANT")
    sep=" "
    sep2="</s>"
    system_template: str = "{system_message}"
    # system_message="A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions."
    system_prompt = system_template.format(system_message=system_message)

    seps = [sep, sep2]
    ret = system_prompt + seps[0]
    prompt, response = None, None
    for i, (role, message) in enumerate(messages):
        if message:
            prompt = ret + role + ":"
            response = " " + message + seps[i % 2]
            ret += role + ": " + message + seps[i % 2]
        else:
            raise AssertionError("Messages not completed.")
            # ret += role + ":"
    return ret, prompt, response


def vicuna_sample_processor(prompt, response, system_message=vicuna_system_prompt):
    _ , vicuna_prompt, response = build_vicuna_input([["USER", prompt], ['ASSISTANT', response]], system_message)
    return vicuna_prompt, response


def print_rank(*args, rank=0, **kwargs):
    if not dist.is_initialized() or dist.get_rank() == rank:
        print(*args, **kwargs)


def broadcast_model(model, rank=0):
    """
    Broadcasts the `model` to all other processes.
    """
    if not dist.is_initialized():
        return
    for p in model.parameters():
        dist.broadcast(p, src=rank)


def move_dict_to_device(d: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    """
    Moves a dict of tensors to a given device.
    """
    for k, v in d.items():
        d[k] = v.to(device)
    return d


def save_checkpoint(save_path: str, model: PreTrainedModel, tokenizer: PreTrainedTokenizer, checkpoint_info: Optional[Dict] = None):
    os.makedirs(save_path, exist_ok=True)
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    if checkpoint_info is not None:
        with open(os.path.join(save_path, "checkpoint_info.json"), "w") as f:
            json.dump(checkpoint_info, f)