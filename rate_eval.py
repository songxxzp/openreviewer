import json
import numpy as np

rating = 10
soundness = 4
contribution = 4
confidence = 4
presentation = 4

def compute_average(datas):
    rating_loss, soundness_loss, presentation_loss, contribution_loss, confidence_loss = [], [], [], [], []
    print(len(datas))
    for i in range(len(datas)):
        rating_loss.append((datas[i]["Rating"]["Mean"] - datas[i]["Rating"]["Label Mean"]) / rating)
        soundness_loss.append((datas[i]["Soundness"]["Mean"] - datas[i]["Soundness"]["Label Mean"]) / soundness)
        presentation_loss.append((datas[i]["Presentation"]["Mean"] - datas[i]["Presentation"]["Label Mean"]) / presentation)
        contribution_loss.append((datas[i]["Contribution"]["Mean"] - datas[i]["Contribution"]["Label Mean"]) / contribution)
        confidence_loss.append((datas[i]["Confidence"]["Mean"] - datas[i]["Confidence"]["Label Mean"]) / confidence)
    rating_average = np.mean(np.array(rating_loss))
    soundness_average = np.mean(np.array(soundness_loss))
    presentation_average = np.mean(np.array(presentation_loss))
    contribution_average = np.mean(np.array(contribution_loss))
    confidence_average = np.mean(np.array(confidence_loss))
    rating_average_abs = np.mean(abs(np.array(rating_loss)))
    soundness_average_abs = np.mean(abs(np.array(soundness_loss)))
    presentation_average_abs = np.mean(abs(np.array(presentation_loss)))
    contribution_average_abs = np.mean(abs(np.array(contribution_loss)))
    confidence_average_abs = np.mean(abs(np.array(confidence_loss)))

    

    return rating_average, soundness_average, presentation_average, contribution_average, confidence_average, rating_average_abs, soundness_average_abs, presentation_average_abs, contribution_average_abs, confidence_average_abs

data_path = "/root/autodl-tmp/workspace/openreviewer/baseline_eval.json"
# data_path = "2023-testset/processed-0101-merge-2048-matched-cleaned-test-v2-baseline-in-agent-format-result-eval.jsonl"
# data_path = "2023-testset/processed-0101-merge-2048-matched-cleaned-test-v2-sft-baseline-result-eval.jsonl"
# data_path = "reviewer_eval.json"
# data_path = "2023-testset/processed-0101-merge-2048-matched-cleaned-test-v2-agent-result-eval.jsonl"
# data_path = "2023-testset/processed-0101-merge-2048-matched-cleaned-test-v2-vanilla-agent-format-fewshot-result-4-vllm-eval.jsonl"
# data_path = "2023-testset/processed-0101-merge-2048-matched-cleaned-test-v2-vanilla-baseline-format-fewshot-result-4-vllm-eval.jsonl"

with open(data_path, 'r', encoding='utf-8') as f:
    datas = json.load(f)
    rating_average, soundness_average, presentation_average, contribution_average, confidence_average, rating_average_abs, soundness_average_abs, presentation_average_abs, contribution_average_abs, confidence_average_abs= compute_average(datas)
    print("rating:"+ f"""{rating_average}""")
    print("soundness:" + f"""{soundness_average}""")
    print("presentation:" + f"""{presentation_average}""")
    print("contribution:" + f"""{contribution_average}""")
    print("confidence:" + f"""{confidence_average}""")
    print("rating:"+ f"""{rating_average_abs}""")
    print("soundness:" + f"""{soundness_average_abs}""")
    print("presentation:" + f"""{presentation_average_abs}""")
    print("contribution:" + f"""{contribution_average_abs}""")
    print("confidence:" + f"""{confidence_average_abs}""")

if "ReviewsBLEU" in datas[0]:
    ReviewsBLEU = sum([sample["Reviews BLEU"] for sample in datas]) / len(datas)
    print("ReviewsBLEU", ReviewsBLEU)

if "Winrate" in datas[0]:
    Winrate = sum([sum(sample["Winrate"]) for sample in datas]) / sum([len(sample["Winrate"]) for sample in datas])
    print("Winrate", Winrate)