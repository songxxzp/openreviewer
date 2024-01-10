import torch
import vllm
import os
import re
import json
from openreviewer.common import vicuna_system_prompt
from openreviewer.utils import build_vicuna_input


model_path = "/root/autodl-tmp/model/vicuna-7b-v1.5-16k"
gpu_memory_utilization = 0.8

llm = vllm.LLM(model=model_path, tensor_parallel_size=torch.cuda.device_count(), gpu_memory_utilization=gpu_memory_utilization)

def get_divided_prompt(text, category):
    messages = [
        ["USER", "The input is the"+category+"part of a review of a paper. "+"Hierarchically analyze this input passage after 'Input:' and provide a breakdown using points 1, 2, 3, and so on. Your output shouldn't add or delete any words comparing with the input. You only need to add breakdown numbers to the iuput. /n For example,  if my input is 'This paper proposed an end-to-end general-purpose any-to-any MM-LLM system, NExT-GPT, by connecting an LLM with multimodal adaptors and different diffusion decoders.', your output should be: '1.This paper proposed an end-to-end general-purpose any-to-any MM-LLM system, NExT-GPT /n 2.by connecting an LLM with multimodal adaptors and different diffusion decoders. /n'. /n Input:/n" + text["value"]],
        ["ASSISTANT", "This part is not used when producing prompt."]
    ]

    _, prompt, _ = build_vicuna_input(messages, vicuna_system_prompt)
    return prompt


def get_divided_responses(prompts):
    sampling_params = vllm.SamplingParams(
        n=1,  # num samples
        temperature=0.7,
        max_tokens=4096
    )
    outputs = llm.generate(prompts, sampling_params)
    responses = [output.outputs[0].text for output in outputs]
    return responses


def get_divided_text(reponse):
    split_points = re.split(r'\d+\.', reponse)[1:]

    result_list = [point.strip() for point in split_points if point.strip()]
    
    return result_list


def reform_review(review):
    reformed_review = dict()

    prompts = [get_divided_prompt(review["summary"], "summary"), get_divided_prompt(review["strengths"], "strengths"), get_divided_prompt(review["weaknesses"], "weaknesses"), get_divided_prompt(review["questions"], "questions")]
    responses = get_divided_responses(prompts)

    reformed_review["summary"] = get_divided_text(responses[0])
    reformed_review["strengths"] = get_divided_text(responses[1])
    reformed_review["weaknesses"] = get_divided_text(responses[2])
    reformed_review["questions"] = get_divided_text(responses[3])
    reformed_review["soundness"] = review["soundness"]["value"]
    reformed_review["presentation"] = review["presentation"]["value"]
    reformed_review["rating"] = review["rating"]["value"]
    reformed_review["contribution"] = review["contribution"]["value"]
    reformed_review["confidence"] = review["confidence"]["value"]
    
    return reformed_review
    

def extract_reviews_from_json(jsonString):
    reviews = json.loads(jsonString)
    data = []  
    for i in range(0, len(reviews)):
        data.append({"Reviews": reform_review(reviews[i]["content"])})

    return data 

dir = '/root/autodl-tmp/workspace/openreviewer/data/iclr2024/reviews/'
another_target_dir = '/root/autodl-tmp/workspace/openreviewer/data/iclr2024/reviews_2/'
target_dir = '/root/autodl-tmp/workspace/openreviewer/data/iclr2024/reviews_4/'

os.makedirs(target_dir, exist_ok=True)
entries = os.listdir(dir)

for idx, entry in enumerate(entries):
    if os.path.exists(another_target_dir + entry) == False :
        print(idx, '/', len(entries), entry)
        with open(dir + entry, "r", encoding='utf-8') as f:
            data = extract_reviews_from_json(f.read())
            with open(target_dir + entry, "w") as write:
                json.dump(data, write)