import torch
import vllm
import os
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
        ["USER", "Prompt:/n/n Hierarchically analyze the following passage and provide a breakdown using points 1, 2, 3, and so on. Your output should not have much difference compared with input./n/n Input:/n/n" 
            + text["value"]],
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
    
    return outputs[0].outputs[0].text

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
target_dir = '/root/autodl-tmp/workspace/openreviewer/data/iclr2024/reviews_/'
entries = os.listdir(dir)

for entry in entries:
    print(entry)
    with open(dir + entry, "r", encoding='utf-8') as f:
        data = extract_reviews_from_json(f.read())
        with open(target_dir + entry, "w") as write:
            json.dump(data, write)