import json
import os

from tqdm import tqdm
import re

cru = os.path.dirname(__file__)
r_file_path = os.path.join(cru, 'train.json')
w_file_path = os.path.join(cru, 'train_trunc.json')

rf = open(r_file_path, mode='r', encoding='utf-8')
wf = open(w_file_path, mode='w', encoding='utf-8')
data = json.load(rf)
target = []
t_cnt = 0

for i, item in enumerate(tqdm(data)):
    q = item["question"]
    id = item["id"]
    is_impossible = item["is_impossible"]
    answers = item["answers"] # 数组
    context = item["context"]

    if len(context) > 256:
        if is_impossible == False:
            ans_text = answers[0]["text"]
            answer_start = answers[0]["answer_start"]
            trunc_start = int(max(0, answer_start - 128))
            trunc_end = int(min(answer_start + 128, len(context) - 2))
            context = context[trunc_start:trunc_end]
            t_cnt = t_cnt + 1

    element = {
        "question": q,
        "id": id,
        "is_impossible": is_impossible,
        "answers": answers,
        "context": context
    }
    target.append(element)

print('Truncate done, total: %d' % t_cnt)
json.dump(obj=target, fp=wf, indent=4, ensure_ascii=False)
print('Write done')