import sys

sys.path.append('../')
import json
import os

from tqdm import tqdm
from random import sample

cru = os.path.dirname(__file__)
r_file_path = os.path.join(cru, 'train_trunc.json')
w_file_path = os.path.join(cru, 'human_test.json')
order_path = os.path.join(cru, '../random_order.json')

rf = open(r_file_path, mode='r', encoding='utf-8')
wf = open(w_file_path, mode='w', encoding='utf-8')

random_order = json.load(open(order_path))
data = json.load(rf)
valid_data = [data[it] for i, it in enumerate(random_order) if i % 10 == 0]
while True:
    target = sample(valid_data, 100)
    noans_cnt = 0
    for line in target:
        if line["is_impossible"]:
            noans_cnt = noans_cnt + 1
    if noans_cnt <= 10:
        break
    else:
        print("noans data: ", noans_cnt)

print('Sample 100 of trunc data done')
json.dump(obj=target, fp=wf, indent=4, ensure_ascii=False)
print('Write done')
