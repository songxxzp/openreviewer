import json
import random


from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer


random.seed(42)


data_path = "/root/autodl-tmp/data/iclr2024/processed-0101-merge-2048-matched.jsonl"
save_train_path = "/root/autodl-tmp/data/iclr2024/processed-0101-merge-2048-matched-cleaned-train.jsonl"
save_test_path = "/root/autodl-tmp/data/iclr2024/processed-0101-merge-2048-matched-cleaned-test.jsonl"
model_path = "/root/autodl-tmp/model/vicuna-7b-v1.5-16k"


tokenizer = AutoTokenizer.from_pretrained(model_path)


remove_cnt = 0
remove_strs = []

for i in range(0, 10):
    remove_strs.append(f"{i + 1}\n\nUnder review as a conference paper at ICLR 2024\n\n")
remove_strs.append("Under review as a conference paper at ICLR 2024\n\n")
for i in range(0, 10):
    remove_strs.append(f"{i + 1}\n\nUnder review as a conference paper at ICLR 2024")
remove_strs.append("Under review as a conference paper at ICLR 2024")

with open(data_path, 'r', encoding='utf-8') as f:
    samples = [json.loads(l) for l in f]

cnt = 0
new_samples = []
assessments = ['summaries', 'strengths', 'weaknesses', 'questions']

for sample in samples:
    flag = True
    text = ''
    for review in sample['Reviews']:
        for assessment in assessments:
            for idx, (splited_assessment, splited_assessment_matched) in enumerate(zip(review[f'splited_{assessment}'], review[f'splited_{assessment}_matched'])):
                pass
        
    for key in sample['Splited_Text']:
        for remove_str in remove_strs:
            if remove_str in sample['Splited_Text'][key]:
                sample['Splited_Text'][key] = sample['Splited_Text'][key].replace(remove_str, '')
                # print("===")
                # print(remove_str)
                # print("===")
                remove_cnt += 1

        if not sample['Splited_Text'][key]:
            flag = False
        else:
            text += sample['Splited_Text'][key] + '\n\n'
    sample["Cleaned_Text"] = text
    length = len(tokenizer.encode(text))
    # print(length)
    if flag and 512 < length < 1024 * 3:
        new_samples.append(sample)
        cnt += 1
    # break

# print(text)
# print(sample['Splited_Text'])
print(f"remove_cnt = {remove_cnt}")
print(f"samples = {cnt}")

random.shuffle(new_samples)


with open(save_train_path, 'w', encoding='utf-8') as f:
    for sample in new_samples[:-100]:
        f.write(json.dumps(sample, ensure_ascii=False) + '\n')


with open(save_test_path, 'w', encoding='utf-8') as f:
    for sample in new_samples[-100:]:
        f.write(json.dumps(sample, ensure_ascii=False) + '\n')
