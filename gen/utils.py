import os
import json

import numpy as np


def divide_data(all_data, tek=10):
    curr = os.path.dirname(__file__)
    ord_path = os.path.join(curr, 'random_order.json')
    # 保存一个随机序（供划分valid用）
    if not os.path.exists(ord_path):
        random_order = list(range(len(all_data)))
        np.random.shuffle(random_order)
        json.dump(random_order, open(ord_path, 'w'), indent=4)
    else:
        random_order = json.load(open(ord_path))
    # 划分valid
    train_data = [all_data[j]
                  for i, j in enumerate(random_order) if i % tek != 0]
    valid_data = [all_data[j]
                  for i, j in enumerate(random_order) if i % tek == 0]
    return train_data, valid_data
