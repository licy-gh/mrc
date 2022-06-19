import logging
import os
from pathlib import Path
import json
import numpy as np
from tqdm import tqdm
from collections import defaultdict

GEN_ROOT_PATH = os.path.dirname(__file__)

# data
mrc_dataset_path = os.path.join(GEN_ROOT_PATH, "dataset/train_trunc.json")
bert_wwm_pt_path = os.path.join(GEN_ROOT_PATH, "chinese-bert-wwm-ext")
config_path = os.path.join(bert_wwm_pt_path, "config.json")
dict_path = os.path.join(bert_wwm_pt_path, "vocab.txt")
pretrained_model_path = os.path.join(bert_wwm_pt_path, "pytorch_model.bin")
save_model_path = os.path.join(GEN_ROOT_PATH, "finetuned_model/rc_model_gen.pt")
log_dir = os.path.join(GEN_ROOT_PATH, "log")

# 参数
gen_max_p_len = 256
gen_max_q_len = 64
gen_max_a_len = 32
gen_max_qa_len = gen_max_q_len + gen_max_a_len
batch_size = 16
epochs = 10
learning_rate = 1e-5

gradient_accumulation_steps = 1
gen_max_grad_norm = 5.0

###############################################
# log
###############################################

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s %(name)s:%(levelname)s:%(message)s",
    datefmt="%d-%m-%Y %H:%M:%S",
    handlers=[
        logging.FileHandler(
            filename=os.path.join(log_dir, "gen_run.log"),
            mode="w",
            encoding="utf-8"
        )
    ]
)
logger = logging.getLogger(__name__)

logger.info(f'begin gen progress ...')