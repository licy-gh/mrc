import re

import torch
import torch.nn as nn

from bojone_snippets import DataGenerator, sequence_padding
from config import *
from evaluation import eval_valid_data
from model_unilm import BojoneModel
from optimizer import create_optimizer_and_scheduler
from rc import GenAnswer, ReadingComprehension
from utils import divide_data

loss_func = nn.CrossEntropyLoss(reduction="none")


class data_generator(DataGenerator):
    """数据生成器
    """

    def __init__(self, data, batch_size, tokenizer, device):
        super(data_generator, self).__init__(data=data, batch_size=batch_size)
        self.tokenizer = tokenizer
        self.device = device

    def __iter__(self, random=False):
        """单条样本格式：[CLS]篇章[SEP]问题[SEP]答案[SEP]
        """
        batch_token_ids, batch_segment_ids = [], []
        for is_end, D in self.sample(random):
            question = D['question']
            is_impossible = D['is_impossible']
            passage = D['context']
            passage = re.sub(u' |、|；|，', ',', passage)
            answer = 'noans' if is_impossible else D['answers'][0]['text']
            qa_token_ids, qa_segment_ids = self.tokenizer.encode(
                question, answer, maxlen=gen_max_qa_len + 1
            )
            p_token_ids, p_segment_ids = self.tokenizer.encode(
                passage, maxlen=gen_max_p_len
            )
            token_ids = p_token_ids + qa_token_ids[1:]
            segment_ids = p_segment_ids + qa_segment_ids[1:]
            batch_token_ids.append(token_ids)
            batch_segment_ids.append(segment_ids)
            if len(batch_token_ids) == self.batch_size or is_end:
                batch_token_ids = torch.tensor(sequence_padding(batch_token_ids), dtype=torch.long, device=self.device)
                batch_segment_ids = torch.tensor(sequence_padding(batch_segment_ids), dtype=torch.long,
                                                 device=self.device)
                yield batch_token_ids, batch_segment_ids
                batch_token_ids, batch_segment_ids = [], []


def compute_loss(inputs):
    y_true, y_mask, y_pred = inputs
    y_true = y_true[:, 1:].contiguous()  # 目标token_ids
    y_mask = y_mask[:, 1:].contiguous()  # segment_ids，刚好指示了要预测的部分
    y_pred = y_pred[:, :-1, :].contiguous()  # 预测序列，错开一位
    loss = loss_func(y_pred.view(-1, y_pred.size(-1)), y_true.view(-1))
    loss = loss.view(y_pred.size(0), -1)
    loss = torch.sum(loss * y_mask) / torch.sum(y_mask)
    return loss


def train_and_eval():
    # 标注数据
    data_f = open(mrc_dataset_path, mode='r', encoding='utf-8')
    all_data = json.load(data_f)

    train_data, valid_data = divide_data(all_data)

    gen = GenAnswer(train_flag=True)

    # train
    tmp_state_dict = torch.load(pretrained_model_path, map_location="cpu")
    tmp_state_dict['bert.embeddings.word_embeddings.weight'] = \
        torch.index_select(tmp_state_dict['bert.embeddings.word_embeddings.weight'],
                           0, torch.tensor(gen.keep_tokens, dtype=torch.long))
    tmp_state_dict["bert.cls.transform.dense.weight"] = \
        tmp_state_dict["cls.predictions.transform.dense.weight"]
    tmp_state_dict["bert.cls.transform.dense.bias"] = \
        tmp_state_dict["cls.predictions.transform.dense.bias"]
    tmp_state_dict["bert.cls.transform.LayerNorm.weight"] = \
        tmp_state_dict["cls.predictions.transform.LayerNorm.weight"]
    tmp_state_dict["bert.cls.transform.LayerNorm.bias"] = \
        tmp_state_dict["cls.predictions.transform.LayerNorm.bias"]

    gen.train_generator = data_generator(data=train_data, batch_size=batch_size, tokenizer=gen.tokenizer,
                                         device=gen.device)

    gen.model = BojoneModel.from_pretrained(
        pretrained_model_name_or_path=bert_wwm_pt_path,
        config=gen.config,
        state_dict=tmp_state_dict
    )
    gen.optimizer, gen.scheduler = create_optimizer_and_scheduler(
        gen.model,
        lr=learning_rate,
        num_training_steps=gen.train_generator.steps * epochs
    )
    gen.model.to(gen.device)
    if torch.cuda.device_count() > 1:
        gen.model = nn.DataParallel(gen.model)

    gen.reader = ReadingComprehension(
        start_id=None,
        end_id=gen.tokenizer._token_end_id,
        maxlen=gen_max_a_len,
        keep_tokens=gen.keep_tokens,
        tokenizer=gen.tokenizer,
        config=gen.config,
        model=gen.model,
        mode='generative'
    )

    best_final = -1
    gen.model.zero_grad()
    for e in tqdm(range(epochs), position=0, leave=False):
        gen.model.train()
        for step, batch in enumerate(tqdm(gen.train_generator, position=1, leave=False)):

            batch = [_.to(gen.device) for _ in batch]
            logits = gen.model(*batch)
            input_ids, segment_ids = batch
            loss = compute_loss((input_ids, segment_ids, logits))

            if gradient_accumulation_steps > 1:
                loss = loss / gradient_accumulation_steps

            loss.backward()

            if (step + 1) % gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(gen.model.parameters(), gen_max_grad_norm)
                gen.optimizer.step()
                gen.scheduler.step()
                gen.optimizer.zero_grad()

            if (step + 1) % 100 == 0:
                logger.info(f"epoch: {e} - step: {step + 1} - loss: {loss}")

        # evaluation
        final = sum(list(eval_valid_data(tmp_valid_data=valid_data[:1000], gen=gen, enum=epochs, show_case=5)))
        if final > best_final:
            best_final = final

            model_to_save = gen.model.module if hasattr(gen.model, "module") else gen.model
            torch.save(model_to_save.state_dict(), save_model_path)


if __name__ == "__main__":
    train_and_eval()
    logger.info('... training process end')
