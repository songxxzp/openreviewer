import openreview
import json, os
import requests
from bs4 import BeautifulSoup
from lxml import html

client = openreview.Client(baseurl='https://api2.openreview.net')

# basic information done
def get_basic_info():
    invitation = 'ICLR.cc/2024/Conference/-/Submission'
    notes = openreview.tools.iterget_notes(client, invitation=invitation)
    with open('data/iclr2024/notes.jsonl', 'w', encoding='utf-8') as f:
        for note in notes:
            note_dict = {
                'cdate': note.cdate,
                'content': note.content,
                'ddate': note.ddate,
                'details': note.details,
                'forum': note.forum,
                'id': note.id,
                'invitation': note.invitation,
                'mdate': note.mdate,
                'nonreaders': note.nonreaders,
                'number': note.number,
                'odate': note.odate,
                'original': note.original,
                'pdate': note.pdate,
                'readers': note.readers,
                'referent': note.referent,
                'replyto': note.replyto,
                'signatures': note.signatures,
                'tcdate': note.tcdate,
                'tmdate': note.tmdate,
                'writers': note.writers
            }
            f.write(json.dumps(note_dict) + '\n')

# get all the pdfs
# 恢复爬取定位
def get_pdfs():
    invitation = 'ICLR.cc/2024/Conference/-/Submission'
    notes = openreview.tools.iterget_notes(client, invitation=invitation)
    pdf_files_count = len([name for name in os.listdir('data/iclr2024/pdfs') if os.path.isfile(os.path.join('data/iclr2024/pdfs', name))])
    print(pdf_files_count)
    # 从 pdf_files_count + 1 开始爬取
    for index, note in enumerate(notes, start=1):
        if index <= pdf_files_count:
            continue  # 跳过

        pdf_url = f'https://api.openreview.net/pdf?id={note.id}'
        response = requests.get(pdf_url)
        print(f"getting the pdf of {note.content['title']} ...")
        if response.status_code == 200:
            with open(f'data/iclr2024/pdfs/{note.id}.pdf', 'wb') as pdf_file:
                pdf_file.write(response.content)

def get_reviewer():
    submissions = client.get_all_notes(invitation="ICLR.cc/2024/Conference/-/Submission", details='directReplies')
    for submission in submissions:
        # print(str(submission.details["directReplies"]))
        reviews = [reply for reply in submission.details["directReplies"] if any(invitation.endswith("Official_Review") for invitation in reply.get("invitations", []))]
        if reviews:  # 确保 reviews 列表不为空
            id = reviews[0]["replyto"]
            with open(f'data/iclr2024/reviews/{id}.json', 'w') as outfile:
                json.dump(reviews, outfile, indent=4)

def get_rebuttals():
    # Fetch all submissions
    submissions = client.get_all_notes(invitation="ICLR.cc/2024/Conference/-/Submission", details='replies')

    for submission in submissions:
        title = submission.content["title"]["value"]
        id = submission.id
        data = submission.details

        reviews = {}  # 存储评论
        review_signatures = {}  # 用于追踪审稿人签名和评论ID
        
        # 遍历所有回复
        for reply in data['replies']:
            reply_id = reply['id']
            reply_to = reply.get('replyto', None)
            signature = reply['signatures'][0]

            if 'Reviewer' in signature:
                if (reply_to is None or reply_to == id):
                    # 这是一轮新的审稿评论
                    reviews[reply_id] = {"review": reply, "responses": []}
                    review_signatures[signature] = reply_id
                else:
                    # 这是对先前审稿评论的后续回复
                    review_id = review_signatures.get(signature)
                    if review_id and review_id in reviews:
                        reviews[review_id]["responses"].append(reply)
            elif 'Authors' in signature and reply_to:
                # 这是作者的回应
                if reply_to in reviews:
                    reviews[reply_to]["responses"].append(reply)

        # 配对审稿人评论与作者回应和后续审稿评论
        paired_reviews_responses = [{"title": title}]
        for review_id, review_data in reviews.items():
            paired_reviews_responses.append(review_data)
        
        with open(f'data/iclr2024/rebuttals/{id}.json', 'w') as outfile:
            json.dump(paired_reviews_responses, outfile, indent=4)
        
        if(title == "SaNN: Simple Yet Powerful Simplicial-aware Neural Networks"): 
            with open(f'id.json', 'w') as outfile:
                json.dump(paired_reviews_responses, outfile, indent=4)

get_rebuttals()

pdf_files_count = len([name for name in os.listdir('data/iclr2024/rebuttals') if os.path.isfile(os.path.join('data/iclr2024/rebuttals', name))])
print(pdf_files_count)
# get_reviewer()
# get_basic_info()
# get_reviewer()
# get_rebuttals()