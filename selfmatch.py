import json

import torch
import torch.nn.functional as F

from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer
from typing import Dict, List, Optional, Tuple


def tokenize_samples(tokenizer: PreTrainedTokenizer, samples: List[Tuple[str, str]]) -> Tuple[List[List[int]], List[List[int]]]:
    # print("samples", samples)
    loss_masks = []
    input_ids = []

    max_len = 0

    for sample in samples:
        prompt, response = sample
        prompt_ids = tokenizer.encode(prompt, add_special_tokens=True)
        response_ids = tokenizer.encode(response, add_special_tokens=False)

        max_len = max(max_len, len(prompt_ids) + len(response_ids))

        input_ids.append(prompt_ids + response_ids)
        loss_masks.append([0] * len(prompt_ids) + [1] * len(response_ids))
    
    # print("max_len:", max_len)

    for i in range(len(samples)):
        loss_masks[i] = [0] * (max_len - len(loss_masks[i])) + loss_masks[i]
        input_ids[i] = [tokenizer.pad_token_id] * (max_len - len(input_ids[i])) + input_ids[i]
    
    return input_ids, loss_masks


def split_to_batch(samples: List[Tuple[str, str]], batch_size: int) -> List[List[Tuple[str, str]]]:
    batches = []
    batch = []
    for sample in samples:
        batch.append(sample)
        if len(batch) == batch_size:
            batches.append(batch)
            batch = []
    if len(batch) > 0:
        batches.append(batch)
    return batches


def select_batch_by_rank(batches: List[Tuple[str, str]], rank: int, world_size: int) -> List[List[Tuple[str, str]]]:
    selected_batches = []
    for idx, batch in enumerate(batches):
        if idx % world_size == rank:
            selected_batches.append(batch)
    return selected_batches


@torch.no_grad()
def compute_loss(model: PreTrainedModel, tokenizer: PreTrainedTokenizer, input_ids: List[List[int]], loss_masks: List[List[int]], device: torch.device, **kwargs):
    temperature = kwargs.get("temperature", 1.0)
    input_ids = torch.tensor(input_ids, device=device, dtype=torch.long)
    loss_masks = torch.tensor(loss_masks, device=device, dtype=torch.long)

    label = input_ids.clone()
    label[(loss_masks == 0)] = -100
    output = model(input_ids, labels=label, use_cache=False)
    logits = output.logits / temperature
    shifted_logits = logits[..., :-1, :]
    shifted_loss_masks = loss_masks[..., 1:]
    shifted_labels = input_ids[..., 1:]
    losses = F.cross_entropy(shifted_logits.reshape(-1, shifted_logits.size(-1)), shifted_labels.reshape(-1, ), reduction='none').reshape(shifted_labels.shape)
    loss = (shifted_loss_masks * losses).sum() / shifted_loss_masks.sum()
    response_logprobs = - (shifted_loss_masks * losses).sum(dim=-1)
    avg_response_logprobs = - (shifted_loss_masks * losses).sum(dim=-1) / shifted_loss_masks.sum(dim=-1)

    # print(f"response_logprobs {response_logprobs}")
    # print(f"avg_response_logprobs {avg_response_logprobs}")
    # print(f"output.loss {output.loss}, loss {loss}")
    # print(f"")
    
    # response_probs = torch.exp(response_logprobs)

    return response_logprobs, avg_response_logprobs


def match_prompt_with_response(model: PreTrainedModel, tokenizer: PreTrainedTokenizer, prompts: List[str], responses: List[str], device: torch.device, **kwargs):
    batch_size = kwargs.get("batch_size", 8)
    rank = kwargs.get("rank", 0)
    world_size = kwargs.get("world_size", 1)

    samples = []

    for prompt in prompts:
        for response in responses:
            sample = (prompt, response)
            samples.append(sample)
    
    all_response_logprobs, all_avg_response_logprobs = [], []

    for batch in split_to_batch(samples, batch_size):
        input_ids, loss_masks = tokenize_samples(tokenizer, batch)
        response_logprobs, avg_response_logprobs = compute_loss(model, tokenizer, input_ids, loss_masks, device, **kwargs)
        all_response_logprobs += response_logprobs.tolist()
        all_avg_response_logprobs += avg_response_logprobs.tolist()


    results = {}
    cnt = 0
    for prompt in prompts:
        results[prompt] = {}
        for response in responses:
            results[prompt][response] = (all_response_logprobs[cnt], all_avg_response_logprobs[cnt])
            cnt += 1
    
    matched_results = {}
    for response in responses:
        max_prob_prompt = None
        for prompt in prompts:
            if max_prob_prompt is None or results[prompt][response][0] > results[max_prob_prompt][response][0]:
                max_prob_prompt = prompt
        matched_results[response] = max_prob_prompt

    return results, matched_results


def build_vicuna_input(prompt):
    messages = [["USER", prompt], ["ASSISTANT", "None"]]
    # roles=("USER", "ASSISTANT")
    sep=" "
    sep2="</s>"
    system_template: str = "{system_message}"
    system_message="A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions."
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
    return prompt


def build_match_prompt(section, section_name, review_key):
    prompt = f"Read the following {section_name}, and write {review_key} for it:\n\n{section}\n\nAfter reading the above {section_name}, write {review_key} for it."
    prompt = build_vicuna_input(prompt)
    return prompt


def filter_prompt(prompts, tokenizer, max_length=4096):
    long_prompts = [prompt for prompt in prompts if len(tokenizer.encode(prompt)) >= max_length]
    prompts = [prompt for prompt in prompts if len(tokenizer.encode(prompt)) < max_length]
    return prompts, long_prompts


def main():
    # data_path = "/root/autodl-tmp/workspace/openreviewer/example-1231.jsonl"
    # save_path = "/root/autodl-tmp/workspace/openreviewer/example-1231-matched.jsonl"

    # data_path = "/root/autodl-tmp/data/iclr2024/processed-1231.jsonl"
    # save_path = "/root/autodl-tmp/data/iclr2024/processed-1231-matched.jsonl"

    data_path = "/root/autodl-tmp/data/iclr2024/processed-1231-merge-2048.jsonl"
    save_path = "/root/autodl-tmp/data/iclr2024/processed-1231-merge-2048-matched.jsonl"

    model_path = "/root/autodl-tmp/model/vicuna-7b-v1.5-16k"
    device = 'cuda'

    with open(data_path, 'r', encoding='utf-8') as f:
        samples = [json.loads(line) for line in f]

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path)

    model.to(device)
    model.to(torch.bfloat16)
    model.eval()

    for idx, sample in enumerate(samples):
        sections = sample["Splited_Text"]
        for review in sample["Reviews"]:
            review_keys = ["summaries", "questions", "weaknesses", "strengths"]
            for review_key in review_keys:
                review["splited_" + review_key + "_matched"] = []
                review_values = review["splited_" + review_key]
                prompts = [build_match_prompt(section, section_name, review_key) for section_name, section in sections.items()]
                prompts, long_prompts = filter_prompt(prompts, tokenizer)
                if len(long_prompts) > 0:
                    # print("len(long_prompts) =", len(long_prompts))
                    raise AssertionError(f"len(long_prompts) = {len(long_prompts)}")
                
                responses = [review_value for review_value in review_values]
                    
                results, matched_results = match_prompt_with_response(
                    model,
                    tokenizer,
                    prompts,
                    responses,
                    device=device,
                    temperture=1.0,
                    batch_size=2
                )

                # print(matched_results)

                for response in responses:
                    max_prob_prompt = None
                    for prompt in prompts:
                        if max_prob_prompt is None or results[prompt][response][0] > results[max_prob_prompt][response][0]:
                            max_prob_prompt = prompt
                    if max_prob_prompt is not None:
                        prompt = max_prob_prompt.split("USER: ")[1].split('ASSISTANT:')[0]
                        review["splited_" + review_key + "_matched"].append(prompt.split('Read the following ')[1].split(f', and write {review_key} for it:\n\n')[0])
                    else:
                        review["splited_" + review_key + "_matched"].append(None)
                    # print(f"Response: {response} | Prompt: {prompt}")
                print(idx, review["splited_" + review_key])
                print(idx, review["splited_" + review_key + "_matched"])

        with open(save_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(sample) + '\n')

    with open(save_path, 'w', encoding='utf-8') as f:
        for sample in samples:
            f.write(json.dumps(sample) + '\n')

if __name__ == "__main__":
    main()

