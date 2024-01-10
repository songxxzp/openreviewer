import torch
import vllm
import json
import re
import nltk
import numpy as np
import sys
from nltk.translate.bleu_score import corpus_bleu
from nltk.translate.bleu_score import SmoothingFunction
from openreviewer.common import vicuna_system_prompt
from openreviewer.utils import build_vicuna_input

# 下载 punkt 数据
# nltk.download('punkt')

def get_data(reviews):
    all_soundness, all_presentation, all_rating, all_contribution, all_confidence = [], [], [], [], []
    for i in range(len(reviews["Reviews"])):
        if(re.search(r"\d", reviews["Reviews"][i]["soundness"])):
            soundness_data = float(re.findall(r"\d+", reviews["Reviews"][i]["soundness"])[0])
        else: 
            soundness_data = 2.5
        if(re.search(r"\d", reviews["Reviews"][i]["presentation"])):
            presentation_data = float(re.findall(r"\d+", reviews["Reviews"][i]["presentation"])[0])
        else:
            presentation_data = 2.5
        if(re.search(r"\d", reviews["Reviews"][i]["rating"])):
            rating_data = float(re.findall(r"\d+", reviews["Reviews"][i]["rating"])[0])
        else:
            rating_data = 5.5
        if(re.search(r"\d", reviews["Reviews"][i]["contribution"])):
            contribution_data = float(re.findall(r"\d+", reviews["Reviews"][i]["contribution"])[0])
        else:
            contribution_data = 2.5
        if(re.search(r"\d", reviews["Reviews"][i]["confidence"])):
            confidence_data = float(re.findall(r"\d+", reviews["Reviews"][i]["confidence"])[0])
        else:
            confidence_data = 2.5
        all_soundness.append(soundness_data)
        all_presentation.append(presentation_data)
        all_rating.append(rating_data)
        all_contribution.append(contribution_data)
        all_confidence.append(confidence_data)
    return all_soundness, all_presentation, all_rating, all_contribution, all_confidence

def data_eval(reviews):
    # 获取数据
    all_soundness, all_presentation, all_rating, all_contribution, all_confidence = get_data(reviews)
    # 计算平均值
    soundness_mean = np.mean(all_soundness)
    presentation_mean = np.mean(all_presentation)
    rating_mean = np.mean(all_rating)
    contribution_mean = np.mean(all_contribution)
    confidence_mean = np.mean(all_confidence)
    # 
    soundness_var = np.var(all_soundness)
    presentation_var = np.var(all_presentation)
    rating_var = np.var(all_rating)
    contribution_var = np.var(all_contribution)
    confidence_var = np.var(all_confidence)
    return soundness_mean, presentation_mean, rating_mean, contribution_mean, confidence_mean, soundness_var, presentation_var, rating_var, contribution_var, confidence_var

# def compute_bleu(outputs, labels):
    label_summary, label_strengths, label_weaknesses, label_questions = [], [], [], []
    summary_bleu, strengths_bleu, weaknesses_bleu, questions_bleu = [], [], [], []
    review_bleu = []
    for i in range(len(labels["Reviews"])):
        label_summary.append(labels["Reviews"][i]["summary"])
        label_strengths.append(labels["Reviews"][i]["strengths"])
        label_weaknesses.append(labels["Reviews"][i]["weaknesses"])
        label_questions.append(labels["Reviews"][i]["questions"])
    reference_tokens_summary = [nltk.word_tokenize(ref.lower()) for ref in label_summary]
    reference_tokens_strengths = [nltk.word_tokenize(ref.lower()) for ref in label_strengths]
    reference_tokens_weaknesses = [nltk.word_tokenize(ref.lower()) for ref in label_weaknesses]
    reference_tokens_questions = [nltk.word_tokenize(ref.lower()) for ref in label_questions]

    for i in range(len(outputs["Reviews"])):
        candidate_tokens_summary = nltk.word_tokenize(outputs["Reviews"][i]["summary"].lower())
        candidate_tokens_strengths = nltk.word_tokenize(outputs["Reviews"][i]["strengths"].lower())
        candidate_tokens_weaknesses = nltk.word_tokenize(outputs["Reviews"][i]["weaknesses"].lower())
        candidate_tokens_questions = nltk.word_tokenize(outputs["Reviews"][i]["questions"].lower())

        summary_bleu.append(corpus_bleu([reference_tokens_summary], [candidate_tokens_summary], smoothing_function=SmoothingFunction().method1))
        strengths_bleu.append(corpus_bleu([reference_tokens_strengths], [candidate_tokens_strengths], smoothing_function=SmoothingFunction().method1))
        weaknesses_bleu.append(corpus_bleu([reference_tokens_weaknesses], [candidate_tokens_weaknesses], smoothing_function=SmoothingFunction().method1))
        questions_bleu.append(corpus_bleu([reference_tokens_questions], [candidate_tokens_questions], smoothing_function=SmoothingFunction().method1))

    for i in range(len(outputs["Reviews"])):
        review_bleu.append(0.25 * (summary_bleu[i] + strengths_bleu[i] + weaknesses_bleu[i] + questions_bleu[i]))  
    
    return review_bleu

# def compute_bleu(labels):
#     label_summary, label_strengths, label_weaknesses, label_questions = [], [], [], []
#     self_summary_bleu, self_questions_bleu, self_strength_bleu, self_weakness_bleu = [], [], [], []
    # for i in range(len(labels["Reviews"])):
    #     label_summary.append(labels["Reviews"][i]["summary"])
    #     label_strengths.append(labels["Reviews"][i]["strengths"])
    #     label_weaknesses.append(labels["Reviews"][i]["weaknesses"])
    #     label_questions.append(labels["Reviews"][i]["questions"])
    # reference_tokens_summary = [nltk.word_tokenize(ref.lower()) for ref in label_summary]
    # reference_tokens_strengths = [nltk.word_tokenize(ref.lower()) for ref in label_strengths]
    # reference_tokens_weaknesses = [nltk.word_tokenize(ref.lower()) for ref in label_weaknesses]
    # reference_tokens_questions = [nltk.word_tokenize(ref.lower()) for ref in label_questions]
    
    # for i in range(len(labels["Reviews"])):
    #     for j in range(i+1, len(labels["Reviews"])):
    #         self_summary_bleu.append(corpus_bleu([reference_tokens_summary[i]], [reference_tokens_summary[j]], smoothing_function=SmoothingFunction().method1))
    #         self_strength_bleu.append(corpus_bleu([reference_tokens_strengths[i]], [reference_tokens_strengths[j]], smoothing_function=SmoothingFunction().method1))
    #         self_weakness_bleu.append(corpus_bleu([reference_tokens_weaknesses[i]], [reference_tokens_weaknesses[j]], smoothing_function=SmoothingFunction().method1))
    #         self_questions_bleu.append(corpus_bleu([reference_tokens_questions[i]], [reference_tokens_questions[j]], smoothing_function=SmoothingFunction().method1))
def compute_bleu(labels):
    label_summary, label_strengths, label_weaknesses, label_questions = [], [], [], []
    self_summary_bleu, self_questions_bleu, self_strength_bleu, self_weakness_bleu = [], [], [], []
    
    # 检查字典中是否包含 "Reviews" 键
    if "Reviews" in labels:
        for i in range(len(labels["Reviews"])):
            # 检查字典中是否包含 "summary"、"strengths"、"weaknesses" 和 "questions" 键
            if "summary" in labels["Reviews"][i]:
                label_summary.append(labels["Reviews"][i]["summary"])
            if "strengths" in labels["Reviews"][i]:
                label_strengths.append(labels["Reviews"][i]["strengths"])
            if "weaknesses" in labels["Reviews"][i]:
                label_weaknesses.append(labels["Reviews"][i]["weaknesses"])
            if "questions" in labels["Reviews"][i]:
                label_questions.append(labels["Reviews"][i]["questions"])
        
        reference_tokens_summary = [nltk.word_tokenize(ref.lower()) for ref in label_summary]
        reference_tokens_strengths = [nltk.word_tokenize(ref.lower()) for ref in label_strengths]
        reference_tokens_weaknesses = [nltk.word_tokenize(ref.lower()) for ref in label_weaknesses]
        reference_tokens_questions = [nltk.word_tokenize(ref.lower()) for ref in label_questions]
        
        for i in range(len(labels["Reviews"])):
            for j in range(i+1, len(labels["Reviews"])):
                # 在使用字典索引之前确保相关键存在
                if i < len(reference_tokens_summary) and j < len(reference_tokens_summary):
                    self_summary_bleu.append(corpus_bleu([[reference_tokens_summary[j]]], [reference_tokens_summary[i]], smoothing_function=SmoothingFunction().method1))
                    self_summary_bleu.append(corpus_bleu([[reference_tokens_summary[i]]], [reference_tokens_summary[j]], smoothing_function=SmoothingFunction().method1))
                if i < len(reference_tokens_strengths) and j < len(reference_tokens_strengths):
                    self_strength_bleu.append(corpus_bleu([[reference_tokens_strengths[j]]], [reference_tokens_strengths[i]], smoothing_function=SmoothingFunction().method1))
                    self_strength_bleu.append(corpus_bleu([[reference_tokens_strengths[i]]], [reference_tokens_strengths[j]], smoothing_function=SmoothingFunction().method1))
                if i < len(reference_tokens_weaknesses) and j < len(reference_tokens_weaknesses):
                    self_weakness_bleu.append(corpus_bleu([[reference_tokens_weaknesses[j]]], [reference_tokens_weaknesses[i]], smoothing_function=SmoothingFunction().method1))
                    self_weakness_bleu.append(corpus_bleu([[reference_tokens_weaknesses[i]]], [reference_tokens_weaknesses[j]], smoothing_function=SmoothingFunction().method1))
                if i < len(reference_tokens_questions) and j < len(reference_tokens_questions):
                    self_questions_bleu.append(corpus_bleu([[reference_tokens_questions[j]]], [reference_tokens_questions[i]], smoothing_function=SmoothingFunction().method1))
                    self_questions_bleu.append(corpus_bleu([[reference_tokens_questions[i]]], [reference_tokens_questions[j]], smoothing_function=SmoothingFunction().method1))

    summary_bleu = np.mean(np.array(self_summary_bleu))
    strength_bleu = np.mean(np.array(self_strength_bleu))
    weakness_bleu = np.mean(np.array(self_weakness_bleu))
    question_bleu = np.mean(np.array(self_questions_bleu))

    review_bleu = 0.25 * (summary_bleu + strength_bleu + weakness_bleu + question_bleu)
    return review_bleu

model_path = "/root/autodl-tmp/model/vicuna-7b-v1.5-16k"
gpu_memory_utilization = 0.8

# llm = vllm.LLM(model=model_path, tensor_parallel_size=torch.cuda.device_count(), gpu_memory_utilization=gpu_memory_utilization)

def get_contrast(text, ref):
    Instruction = "Give a review of a paper from four aspects 'summary', 'strengths', 'weaknesses' and 'questions'"
    messages = [
        ["USER", 
         f"""Select the output (a) or (b) that best matches the given instruction. Choose your preferred output, which can be subjective. Your answer should ONLY contain: Output (a) or Output (b). Here's an example:

        # Example:
        ## Instruction:
        Give a description of the following job: "ophthalmologist"

        ## Output (a):
        An ophthalmologist is a medical doctor who specializes in the diagnosis and treatment of eye diseases and conditions.

        ## Output (b):
        An ophthalmologist is a medical doctor who pokes and prods at your eyes while asking you to read letters from a chart.

        ## Which is best, Output (a) or Output (b)?
        Output (a)

        Here the answer is Output (a) because it provides a comprehensive and accurate description of the job of an ophthalmologist. In contrast, output (b) is more of a joke.

        # Task:
        Now is the real task, do not explain your answer, just say Output (a) or Output (b).

        ## Instruction:
        {Instruction}

        ## Output (a):
        {text}

        ## Output (b):
        {ref}

        ## Which is best, Output (a) or Output (b)?"""],
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
    if ("(a)" in outputs):
        return True
    if ("(b)" in outputs):
        return False


def compute_winrate(reviews, labels):
    win, winnumber = [], []
    winrate = []
    test_text, label_text = [], []
    for i in range(len(labels["Reviews"])):
        label_text.append("summary:" + labels["Reviews"][i]["summary"] + "\n" + "strengths:" + labels["Reviews"][i]["strengths"] + "\n"
                          + "weaknesses:" + labels["Reviews"][i]["weaknesses"] + "\n" + "questions:" + labels["Reviews"][i]["questions"])
    for i in range(len(reviews["Reviews"])):
        test_text.append("summary:" + reviews["Reviews"][i]["summary"] + "\n" + "strengths:" + reviews["Reviews"][i]["strengths"] + "\n"
                          + "weaknesses:" + reviews["Reviews"][i]["weaknesses"] + "\n" + "questions:" + reviews["Reviews"][i]["questions"])
    # print(test_text[0])
    for i in range(len(reviews["Reviews"])):
        row = []
        win.append(row)
        winnumber.append(row)
        for j in range(len(labels["Reviews"])):
            win[i].append(get_contrast(test_text[i], label_text[j]))
            win[i].append(not get_contrast(label_text[j], test_text[i]))
        winnumber[i] = np.array(win[i]).astype(np.int32) 
        winrate.append(np.mean(winnumber[i]))
    
    return winrate
    


def reviewer_eval(output_path, label_path, eval_path):
    data = []
    data_1 = []
    data_2 = []
    reviews = []
    labels = []
    with open(output_path, 'r', encoding='utf-8') as f:
        for line in f:
            reviews.append(json.loads(line))
    with open(label_path, 'r', encoding='utf-8') as f:
        for line in f:
            labels.append(json.loads(line))
    for i in range(len(reviews)):
        row = dict()
        data.append(row)
        soundness_mean, presentation_mean, rating_mean, contribution_mean, confidence_mean, soundness_var, presentation_var, rating_var, contribution_var, confidence_var = data_eval(reviews[i])
        label_soundness_mean, label_presentation_mean, label_rating_mean, label_contribution_mean, label_confidence_mean, label_soundness_var, label_presentation_var, label_rating_var, label_contribution_var, label_confidence_var = data_eval(labels[i])
        data[i]["Soundness"] = {"Mean": soundness_mean, "Variance": soundness_var, "Label Mean": label_soundness_mean, "Label Variance": label_soundness_var}
        data[i]["Presentation"] = {"Mean": presentation_mean, "Variance": presentation_var, "Label Mean": label_presentation_mean, "Label Variance": label_presentation_var}
        data[i]["Rating"] = {"Mean": rating_mean, "Variance": rating_var, "Label Mean": label_rating_mean, "Label Variance": label_rating_var}
        data[i]["Contribution"] = {"Mean": contribution_mean, "Variance": contribution_var, "Label Mean": label_contribution_mean, "Label Variance": label_contribution_var}
        data[i]["Confidence"] = {"Mean": confidence_mean, "Variance": confidence_var, "Label Mean": label_confidence_mean, "Label Variance": label_confidence_var}
        # 计算每个评论的BLEU分数
        reviews_bleu = compute_bleu(reviews[i])
        labels_bleu = compute_bleu(labels[i])
        data[i]["Reviews BLEU"] = reviews_bleu
        data[i]["Labels BLEU"] = labels_bleu
        data_1.append(reviews_bleu)
        data_2.append(labels_bleu)
        # winrate = compute_winrate(reviews[i], labels[i])
        # data[i]["Winrate"] = winrate

    
    print(np.mean(np.array(data_1)))
    print(np.mean(np.array(data_2)))
    
    with open(eval_path, 'w', encoding='utf-8') as w:
        json.dump(data, w)



if __name__ == '__main__':
    # output_path = "test/processed-0101-merge-2048-matched-cleaned-test-v2-agent-result.jsonl"
    # label_path = "/root/autodl-tmp/workspace/openreviewer/test/processed-0101-merge-2048-matched-cleaned-test-v2.jsonl"
    # eval_path = "/root/autodl-tmp/workspace/openreviewer/baseline_eval_.json"
    label_path = "/root/autodl-tmp/workspace/openreviewer/test/processed-0101-merge-2048-matched-cleaned-test-v2.jsonl"
    # output_path = "2023-testset/processed-0101-merge-2048-matched-cleaned-test-v2-vanilla-agent-format-fewshot-result-4-vllm.jsonl"
    # eval_path = "2023-testset/processed-0101-merge-2048-matched-cleaned-test-v2-vanilla-agent-format-fewshot-result-4-vllm-eval.jsonl"
    output_path = "2023-testset/processed-0101-merge-2048-matched-cleaned-test-v2-sft-baseline-result.jsonl"
    eval_path = "2023-testset/processed-0101-merge-2048-matched-cleaned-test-v2-sft-baseline-result-eval-2.jsonl"
    # output_path = "2023-testset/processed-0101-merge-2048-matched-cleaned-test-v2-baseline-in-agent-format-result.jsonl"
    # eval_path = "2023-testset/processed-0101-merge-2048-matched-cleaned-test-v2-baseline-in-agent-format-result-eval.jsonl"
    # output_path = "2023-testset/processed-0101-merge-2048-matched-cleaned-test-v2-agent-result.jsonl"
    # eval_path = "2023-testset/processed-0101-merge-2048-matched-cleaned-test-v2-agent-result-eval.jsonl"
    reviewer_eval(output_path, label_path, eval_path)
