import json
import os

from typing import List, Dict, Any


pdfs_path = "/root/autodl-tmp/data/1204/pdfstxt"
note_path = "/root/autodl-tmp/workspace/openreviewer/data/iclr2024/notes.jsonl"
reviews_path = "/root/autodl-tmp/workspace/openreviewer/data/iclr2024/reviews"
rebuttals_path = "/root/autodl-tmp/workspace/openreviewer/data/iclr2024/rebuttals"
save_path = "/root/autodl-tmp/data/iclr2024/1204.jsonl"


def review_parser(review: Dict, forum: str) -> Dict[str, Any]:
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
    }

    parsed_review['score'] = score_parser(parsed_review)

    return parsed_review


def reviews_parser(reviews: List, forum: str) -> List:
    return [review_parser(review, forum) for review in reviews]


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
    pdfs = {}
    for pdf_path in os.listdir(pdfs_path):
        with open(os.path.join(pdfs_path, pdf_path), "r", encoding="utf-8") as f:
            pdfs[pdf_path.replace(".txt", '')] = f.read()
    reviews = {}
    for review_path in os.listdir(reviews_path):
        with open(os.path.join(reviews_path, review_path), "r", encoding="utf-8") as f:
            reviews[review_path.replace(".json", '')] = json.load(f)
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
        sample = {
            "Title": note["content"]["title"]["value"],
            "Author": None,
            "Abstract": note["content"]["abstract"]["value"],
            "Keywords": note["content"]["keywords"]["value"],
            "Reviews": reviews_parser(reviews[note["forum"]], note["forum"]) if note["forum"] in reviews else None,
            "Text": pdfs[note["forum"]] if note["forum"] in pdfs else None,
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
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w', encoding='utf-8') as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')
    
    print(json.dumps(sample, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

