import openreview
import json, os
import requests

client = openreview.Client(baseurl='https://api.openreview.net')

invitation = 'ICLR.cc/2023/Conference/-/Blind_Submission'

notes = openreview.tools.iterget_notes(client, invitation=invitation)

# basic information done
def get_basic_info():
    with open('notes.jsonl', 'w', encoding='utf-8') as f:
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
    pdf_files_count = len([name for name in os.listdir('data/pdfs') if os.path.isfile(os.path.join('data/pdfs', name))])
    print(pdf_files_count)
    # 从 pdf_files_count + 1 开始爬取
    for index, note in enumerate(notes, start=1):
        if index <= pdf_files_count:
            continue  # 跳过

        pdf_url = f'https://api.openreview.net/pdf?id={note.id}'
        response = requests.get(pdf_url)
        print(f"getting the pdf of {note.content['title']} ...")
        if response.status_code == 200:
            with open(f'data/pdfs/{note.id}.pdf', 'wb') as pdf_file:
                pdf_file.write(response.content)

def get_reviewer():
    submissions = client.get_all_notes(invitation="ICLR.cc/2023/Conference/-/Blind_Submission", details='directReplies')
    reviews = []
    for submission in submissions:
        reviews = [reply for reply in submission.details["directReplies"] if reply["invitation"].endswith("Official_Review")]
        id = reviews[0]["replyto"]
        with open(f'data/reviews/{id}.json', 'w') as outfile:
            json.dump(reviews, outfile, indent=4)

# pdf_files_count = len([name for name in os.listdir('data/reviews') if os.path.isfile(os.path.join('data/reviews', name))])
# print(pdf_files_count)
# get_reviewer()

