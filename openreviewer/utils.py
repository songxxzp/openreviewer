import json
import os
import random
import copy

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


def build_vicuna_input_for_one_turn(prompt, response, system_message=vicuna_system_prompt):
    _ , vicuna_prompt, response = build_vicuna_input([["USER", prompt], ['ASSISTANT', response]], system_message)
    return vicuna_prompt, response


def vicuna_sample_processor(sample: Dict):
    return build_vicuna_input_for_one_turn(**sample)


def build_openreviewer_response(review: Dict) -> str:
    response = f"summary: {review['summary']}\n\nstrengths: {review['strengths']}\n\nweaknesses: {review['weaknesses']}\n\nquestions: {review['questions']}\n\nsoundness: {review['soundness']}\n\npresentation: {review['presentation']}\n\ncontribution: {review['contribution']}\n\nrating: {review['rating']}\n\nconfidence: {review['confidence']}\n\n"
    return response


def build_openreviewer_input(sample: Dict):
    Title = sample['Title']
    Keywords = ", ".join(sample['Keywords'])
    Abstract = sample['Abstract']
    Text = sample['Text']

    system_prompt = f"You are reviewing the paper titled {Title}. The keywords are {Keywords}. You will read this paper and write a review for it."

    if Text is None:
        prompt = f"The abstract is：\n\n{Abstract}\n\nAfter reading this abstract, please write your review for it."
    else:
        prompt = f"The abstract is：\n\n{Abstract}\n\nThe main body of the paper is: \n\n{Text}\n\nAfter reading this paper, please write your review for it."

    if sample['Reviews'] is None:
        responses = []
    else:
        responses = [build_openreviewer_response(review) for review in sample['Reviews']]
    return system_prompt, prompt, responses


def openreviewer_data_preprocessor(samples):
    dataset = []
    for sample in samples:
        system_prompt, prompt, responses = build_openreviewer_input(sample)
        for response in responses:
            dataset.append({"system_message": system_prompt, "prompt": prompt, "response": response})
    return dataset


def reviewer_agent_preprocessor(samples):
    dataset = []
    for sample in samples:
        for review in sample['Reviews']:
            new_sample = copy.copy(sample)
            new_sample['review'] = review
            dataset.append(new_sample)
    return dataset


def reviewer_agent_processor(tokenizer, sample: Dict):
    sections = ['ABSTRACT', 'INTRODUCTION', 'EXPERIMENTS', 'RESULTS', 'CONCLUSION']
    assessments = ['summaries', 'strengths', 'weaknesses', 'questions']

    Title = sample['Title']
    Keywords = ", ".join(sample['Keywords'])
    Abstract = sample['Abstract']
    review = sample['review']
    # dict_keys(['Title', 'Author', 'Abstract', 'Keywords', 'Reviews', 'Text', 'Splited_Text', 'TLDR', 'cdate', 'id', 'forum'])
    # dict_keys(['ABSTRACT', 'INTRODUCTION', 'EXPERIMENTS', 'RESULTS', 'CONCLUSION'])
    # dict_keys(['id', 'forum', 'replyto', 'rating', 'confidence', 'soundness', 'presentation', 'contribution', 'summary', 'strengths', 'weaknesses', 'questions', 'splited_summaries', 'splited_questions', 'splited_weaknesses', 'splited_strengths', 'score', 'splited_summaries_matched', 'splited_questions_matched', 'splited_weaknesses_matched', 'splited_strengths_matched'])

    Seed = random.randint(0, 65535)
    elements = [
        (0, f"You are the No.{Seed} reviewer of openreivew. You are reviewing the paper titled {Title}. The keywords are {Keywords}. You will read this paper and write a review for it.\n\n")
    ]
    assessment_dict = {}
    for assessment in assessments:
        assessment_dict[assessment] = []
    for section in sections:
        elements.append((0, f"Read the {section} of this paper, and write down your reading notes, such as summaries, questions, weaknesses, or strengths:\n\n"))
        elements.append((0, sample['Splited_Text'][section] + '\n\n'))
        elements.append((0, f"Now write down your note for the {section} of this paper, such as summaries, questions, weaknesses, or strengths:\n\n"))

        local_text = ""
        for assessment in assessments:
            flag = False
            for splited_assessment, splited_assessment_matched in zip(review[f'splited_{assessment}'], review[f'splited_{assessment}_matched']):
                if splited_assessment_matched == section:
                    flag = True
                    break
            if not flag:
                continue

            local_text += f"{assessment.upper()}:\n"
            for splited_assessment, splited_assessment_matched in zip(review[f'splited_{assessment}'], review[f'splited_{assessment}_matched']):
                if splited_assessment_matched == section:
                    assessment_dict[assessment].append(splited_assessment)
                    local_text += f"{splited_assessment}\n"
            local_text += '\n'
        elements.append((1, local_text))
    
    for assessment in assessments:
        elements.append((0, f"You will read through your note about {assessment.upper()}, then write down the {assessment.upper()} for the paper. Your note about {assessment.upper()}:\n\n"))
        for splited_assessment in assessment_dict[assessment]:
            elements.append((0, f"{splited_assessment}\n"))
        elements.append((0, f"\n"))
        elements.append((0, f"Your final {assessment.upper()}:\n\n"))
        elements.append((1, f"{review['summary' if assessment == 'summaries' else assessment]}\n\n"))

    elements.append((0, f"Now give this article a score from multiple dimensions:\n\n"))
    elements.append((1, f"soundness: {review['soundness']}\n\npresentation: {review['presentation']}\n\ncontribution: {review['contribution']}\n\nrating: {review['rating']}\n\nconfidence: {review['confidence']}\n\n"))

    input_ids = tokenizer.encode('', add_special_tokens=True)
    loss_mask = [0] * len(input_ids)

    for element in elements:
        io_mark, string = element
        tokens = tokenizer.encode(string, add_special_tokens=False)
        if io_mark == 0:
            loss_mask += [0] * len(tokens)
        elif io_mark == 1:
            tokens.append(tokenizer.eos_token_id)
            loss_mask += [1] * len(tokens)
        input_ids += tokens
    
    return input_ids, loss_mask


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