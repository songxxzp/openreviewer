import torch
import vllm
import os
import re
import json
from openreviewer.common import vicuna_system_prompt
from openreviewer.utils import build_vicuna_input


# messages = [
#     ["USER", "Put your prompt here"],
#     ["ASSISTANT", "This part is not used when producing prompt."]
# ]

# _, prompt, _ = build_vicuna_input(messages, vicuna_system_prompt)

# print(f"prompt:\n{prompt}")

# model_path = "/root/autodl-tmp/model/vicuna-7b-v1.5-16k"
# gpu_memory_utilization = 0.8


# llm = vllm.LLM(model=model_path, tensor_parallel_size=torch.cuda.device_count(), gpu_memory_utilization=gpu_memory_utilization)

# prompts = [prompt]  # you can put all prompts in this list
# sampling_params = vllm.SamplingParams(
#     n=2,  # num samples
#     temperature=0.7,
#     max_tokens=4096
# )

# outputs = llm.generate(prompts, sampling_params)

# all_reponses = [[output.text for output in output.outputs] for output in outputs]

# # len(all_reponses) == len(prompts)
# # len(reponses) == num_samples

# for prompt, reponses in zip(prompts, all_reponses):
#     print(prompt)
#     print(reponses)
#     print()
model_path = "/root/autodl-tmp/model/vicuna-7b-v1.5-16k"
gpu_memory_utilization = 0.8

llm = vllm.LLM(model=model_path, tensor_parallel_size=torch.cuda.device_count(), gpu_memory_utilization=gpu_memory_utilization)

def get_divided_text(text, category):
    messages = [
        ["USER", "The input is the"+category+"part of a review of a paper. "+"Hierarchically analyze this input passage after 'Input:' and provide a breakdown using points 1, 2, 3, and so on. Your output shouldn't add or delete any words comparing with the input. You only need to add breakdown numbers to the iuput. /n For example,  if my input is 'This paper proposed an end-to-end general-purpose any-to-any MM-LLM system, NExT-GPT, by connecting an LLM with multimodal adaptors and different diffusion decoders.', your output should be: '1.This paper proposed an end-to-end general-purpose any-to-any MM-LLM system, NExT-GPT /n 2.by connecting an LLM with multimodal adaptors and different diffusion decoders. /n'. /n Input:/n" + text["value"]],
        ["ASSISTANT", "This part is not used when producing prompt."]
    ]

    _, prompt, _ = build_vicuna_input(messages, vicuna_system_prompt)


    prompts = [prompt]  # you can put all prompts in this list
    sampling_params = vllm.SamplingParams(
        n=1,  # num samples
        temperature=0.7,
        max_tokens=4096
    )
    outputs = llm.generate(prompts, sampling_params)
    outputs = outputs[0].outputs[0].text
    split_points = re.split(r'\d+\.', outputs)[1:]

    result_list = [point.strip() for point in split_points if point.strip()]
    
    return result_list

def reform_review(review):
    reformed_review = dict()

    reformed_review["summary"] = get_divided_text(review["summary"], "summary")
    reformed_review["strengths"] = get_divided_text(review["strengths"], "strengths")
    reformed_review["weaknesses"] = get_divided_text(review["weaknesses"], "weaknesses")
    reformed_review["questions"] = get_divided_text(review["questions"], "questions")
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
target_dir = '/root/autodl-tmp/workspace/openreviewer/data/iclr2024/reviews_2/'
entries = os.listdir(dir)

for entry in entries:
    if os.path.exists(target_dir + entry) == False :
        print(entry)
        with open(dir + entry, "r", encoding='utf-8') as f:
            data = extract_reviews_from_json(f.read())
            with open(target_dir + entry, "w") as write:
                json.dump(data, write)