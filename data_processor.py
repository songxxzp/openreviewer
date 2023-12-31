import json
import os

from typing import List, Dict, Any


# pdfs_path = "/root/autodl-tmp/data/1204/pdfstxt"
pdfs_path = "/root/autodl-tmp/data/iclr2024/pdfstxt"
note_path = "/root/autodl-tmp/workspace/openreviewer/data/iclr2024/notes.jsonl"
reviews_path = "/root/autodl-tmp/workspace/openreviewer/data/iclr2024/reviews"
splited_reviews_path = "/root/autodl-tmp/workspace/openreviewer/data/iclr2024/reviews_2"
rebuttals_path = "/root/autodl-tmp/workspace/openreviewer/data/iclr2024/rebuttals"
# save_path = "/root/autodl-tmp/data/iclr2024/1227.jsonl"
save_path = "/root/autodl-tmp/workspace/openreviewer/example-1227.jsonl"

# pdfs_path = "data/iclr2024/pdfstxt"
# note_path = "data/iclr2024/notes.jsonl"
# reviews_path = "data/iclr2024/reviews"
# rebuttals_path = "data/iclr2024/rebuttals"
# save_path = "data/iclr2024/1204_3k.jsonl"

chapters = {
    "ABSTRACT": True,
    "INTRODUCTION": True,
    "RELATED WORK": False,
    "EXPERIMENTS": True,
    "RESULTS": True,
    "DISCUSSION": False, 
    "CONCLUSION": True, 
    "REFERENCES": False
}


def split_iclr_pdf(content, keep_chapters) -> Dict[str, str]:
    paper_dict = {}
    keys = list(keep_chapters.keys())
    for i in range(len(keep_chapters.keys())):
        if(not keep_chapters[keys[i]]):
            continue
        start_key = keys[i]
        end_key = keys[i + 1] if i + 1 < len(keys) else None
        paper_dict[start_key] = extract_part(content, start_key, end_key)

    return paper_dict


def filter_iclr_pdf(content, keep_chapters) -> str:
    # iclr键值

    paper_dict = {}
    keys = list(keep_chapters.keys())
    for i in range(len(keep_chapters.keys())):
        if(not keep_chapters[keys[i]]):
            continue
        start_key = keys[i]
        end_key = keys[i + 1] if i + 1 < len(keys) else None
        paper_dict[start_key] = extract_part(content, start_key, end_key)

    formatted_string = ""
    for key, value in paper_dict.items():
        formatted_string += f"{key}\n\n{value}\n\n\n"

    return formatted_string


def extract_part(text, start_key, end_key) -> str:
    start_index = text.lower().find(start_key.lower())
    if start_index == -1:
        return ""
    
    if end_key:
        end_index = text.lower().find(end_key.lower(), start_index)
    else:
        end_index = len(text)

    return text[start_index:end_index].strip()


def review_parser(review: Dict, splited_review: Dict, forum: str) -> Dict[str, Any]:
    assert review["replyto"] == forum
    assert review["forum"] == forum

    parsed_review = {
        "id": review["id"],
        "forum": review["forum"],
        "replyto": review["replyto"],
        "rating": review["content"]["rating"]["value"],
        "confidence": review["content"]["confidence"]["value"],
        "soundness": review["content"]["soundness"]["value"],
        "presentation": review["content"]["presentation"]["value"],
        "contribution": review["content"]["contribution"]["value"],
        "summary": review["content"]["summary"]["value"],
        "strengths": review["content"]["strengths"]["value"],
        "weaknesses": review["content"]["weaknesses"]["value"],
        "questions": review["content"]["questions"]["value"],
        "splited_summaries": splited_review["summary"],
        "splited_questions": splited_review["questions"],
        "splited_weaknesses": splited_review["weaknesses"],
        "splited_strengths": splited_review["strengths"],
    }

    parsed_review['score'] = score_parser(parsed_review)

    return parsed_review


def reviews_parser(reviews: List, splited_reviews: List, forum: str) -> List:
    return [review_parser(review, splited_review["Reviews"], forum) for (review, splited_review) in zip(reviews, splited_reviews)]


def score_parser(parsed_review: Dict[str, Any]) -> Dict[str, int]:
    parsed_score = {
        "rating": int(parsed_review["rating"].split(': ')[0]),
        "confidence": int(parsed_review["confidence"].split(': ')[0]),
        "soundness": int(parsed_review["soundness"].split(' ')[0]),
        "presentation": int(parsed_review["presentation"].split(' ')[0]),
        "contribution": int(parsed_review["contribution"].split(' ')[0]),
    }
    return parsed_score


def main():
    with open(note_path, "r", encoding="utf-8") as f:
        notes = [json.loads(l) for l in f]
    
    empty = 0
    pdfs = {}
    for pdf_path in os.listdir(pdfs_path):
        with open(os.path.join(pdfs_path, pdf_path), "r", encoding="utf-8") as f:
            content = f.read()
            if content:
                pdfs[pdf_path.replace(".txt", '')] = content
            else:
                empty += 1
                print(pdf_path)
    print("Empty pdfs:", empty)

    reviews = {}
    for review_path in os.listdir(reviews_path):
        with open(os.path.join(reviews_path, review_path), "r", encoding="utf-8") as f:
            reviews[review_path.replace(".json", '')] = json.load(f)
    splited_reviews = {}
    for splited_review_path in os.listdir(splited_reviews_path):
        with open(os.path.join(splited_reviews_path, splited_review_path), "r", encoding="utf-8") as f:
            splited_reviews[splited_review_path.replace(".json", '')] = json.load(f)
    rebuttals = {}
    if os.path.exists(rebuttals_path):
        for rebuttal_path in os.listdir(rebuttals_path):
            with open(os.path.join(rebuttals_path, rebuttal_path), "r", encoding="utf-8") as f:
                rebuttals[rebuttal_path.replace(".json", '')] = json.load(f)

    print(f"num notes: {len(notes)}")
    print(f"num pdfs: {len(pdfs)}")
    print(f"num reviews: {len(reviews)}")
    print(f"num rebuttals: {len(rebuttals)}")

    samples = []

    for note in notes:
        assert note["id"] == note["forum"]
        if (not note["forum"] in pdfs) or (not note["forum"] in reviews) or (not note["forum"] in splited_reviews):
            print(note["forum"])
            continue
        sample = {
            "Title": note["content"]["title"]["value"],
            "Author": None,
            "Abstract": note["content"]["abstract"]["value"],
            "Keywords": note["content"]["keywords"]["value"],
            "Reviews": reviews_parser(reviews[note["forum"]], splited_reviews[note["forum"]], note["forum"]) if note["forum"] in reviews else None,
            "Text": pdfs[note["forum"]] if note["forum"] in pdfs else None,
            "Splited_Text": split_iclr_pdf(pdfs[note["forum"]], chapters) if note["forum"] in pdfs else None,
            # "Rebuttals": rebuttals[note["forum"]] if note["forum"] in rebuttals else None,
            # "Scores": "List[int](same number to Reviews)",
            # "References": "List[str]",
            # "...": "..."
            "TLDR": note["content"]["TLDR"]["value"] if "TLDR" in note["content"] else None,
            "cdate": note["cdate"],
            "id": note["id"],
            "forum": note["forum"],
        }
        samples.append(sample)
        break
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w', encoding='utf-8') as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')
    
    print(json.dumps(sample, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

