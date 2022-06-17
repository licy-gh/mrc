# -*- coding: utf-8 -*-
import re

import torch
import torch.nn as nn
from transformers import AutoConfig

from bojone_snippets import AutoRegressiveDecoder, sequence_padding
from bojone_tokenizers import load_vocab, Tokenizer
from config import *
from model_unilm import BojoneModel


class ReadingComprehension(AutoRegressiveDecoder):
    """beam search解码来生成答案
    passages为多篇章组成的list，从多篇文章中自动决策出最优的答案，
    如果没答案，则返回空字符串。
    mode是extractive时，按照抽取式执行，即答案必须是原篇章的一个片段。
    """

    def __init__(
            self,
            tokenizer,
            keep_tokens,
            config,
            model,
            mode='generative',
            **kwargs
    ):
        super(ReadingComprehension, self).__init__(**kwargs)
        self.mode = mode
        self.keep_tokens = keep_tokens
        self.tokenizer = tokenizer
        self.config = config
        self.model = model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        if torch.cuda.device_count() > 1:
            self.model = nn.DataParallel(self.model)

    def get_ngram_set(self, x, n):
        """生成ngram合集，返回结果格式是:
        {(n-1)-gram: set([n-gram的第n个字集合])}
        """
        result = {}
        for i in range(len(x) - n + 1):
            k = tuple(x[i:i + n])
            if k[:-1] not in result:
                result[k[:-1]] = set()
            result[k[:-1]].add(k[-1])
        return result

    @AutoRegressiveDecoder.wraps(default_rtype='probas', use_states=True)
    def predict(self, inputs, output_ids, states):
        inputs = [i for i in inputs if i[0, 0] > -1]  # 过滤掉无答案篇章
        topk = len(inputs[0])
        all_token_ids, all_segment_ids = [], []
        for token_ids in inputs:  # inputs里每个元素都代表一个篇章
            token_ids = np.concatenate([token_ids, output_ids], 1)
            segment_ids = np.zeros_like(token_ids)
            if states > 0:
                segment_ids[:, -output_ids.shape[1]:] = 1
            all_token_ids.extend(token_ids)
            all_segment_ids.extend(segment_ids)
        padded_all_token_ids = torch.tensor(sequence_padding(all_token_ids), dtype=torch.long, device=self.device)
        padded_all_segment_ids = torch.tensor(sequence_padding(all_segment_ids), dtype=torch.long, device=self.device)
        with torch.no_grad():
            probas = self.model(padded_all_token_ids, padded_all_segment_ids)
            probas = torch.softmax(probas, dim=-1)
        probas = probas.cpu().detach().numpy()
        probas = [
            probas[i, len(ids) - 1] for i, ids in enumerate(all_token_ids)
        ]
        probas = np.array(probas).reshape((len(inputs), topk, -1))
        if states == 0:
            # 这一步主要是排除没有答案的篇章
            # 如果一开始最大值就为end_id，那说明该篇章没有答案
            argmax = probas[:, 0].argmax(axis=1)
            available_idxs = np.where(argmax != self.end_id)[0]
            if len(available_idxs) == 0:  # 所有篇章最大值都是end_id
                scores = np.zeros_like(probas[0])
                scores[:, self.end_id] = 1
                return scores, states + 1
            else:
                for i in np.where(argmax == self.end_id)[0]:
                    inputs[i][:, 0] = -1  # 无答案篇章首位标记为-1
                probas = probas[available_idxs]
                inputs = [i for i in inputs if i[0, 0] > -1]  # 过滤掉无答案篇章
        if self.mode == 'extractive':
            # 如果是抽取式，那么答案必须是篇章的一个片段
            # 那么将非篇章片段的概率值全部置0
            new_probas = np.zeros_like(probas)
            ngrams = {}
            for token_ids in inputs:
                token_ids = token_ids[0]  # [1,s] -> [s]
                sep_idx = np.where(token_ids == self.tokenizer._token_end_id)[0][0]
                p_token_ids = token_ids[1:sep_idx]
                for k, v in self.get_ngram_set(p_token_ids, states + 1).items():
                    ngrams[k] = ngrams.get(k, set()) | v
            for i, ids in enumerate(output_ids):
                available_idxs = ngrams.get(tuple(ids), set())
                available_idxs.add(self.tokenizer._token_end_id)
                available_idxs = list(available_idxs)
                new_probas[:, i, available_idxs] = probas[:, i, available_idxs]
            probas = new_probas
        return (probas ** 2).sum(0) / (probas.sum(0) + 1), states + 1  # 某种平均投票方式

    def answer(self, question, passages, topk=1):
        token_ids = []
        for passage in passages:
            passage = re.sub(u' |、|；|，', ',', passage)
            p_token_ids = self.tokenizer.encode(passage, maxlen=gen_max_p_len)[0]
            q_token_ids = self.tokenizer.encode(question, maxlen=gen_max_q_len + 1)[0]
            token_ids.append(p_token_ids + q_token_ids[1:])
        output_ids = self.beam_search(
            token_ids, topk=topk, states=0
        )  # 基于beam search
        return self.tokenizer.decode(output_ids)


class GenAnswer:
    def __init__(self):
        self.token_dict, self.keep_tokens = load_vocab(
            dict_path=dict_path,
            simplified=True,
            startswith=['[PAD]', '[UNK]', '[CLS]', '[SEP]'],
        )
        self.tokenizer = Tokenizer(self.token_dict, do_lower_case=True)
        self.config = AutoConfig.from_pretrained(
            pretrained_model_name_or_path=config_path,
            keep_tokens=self.keep_tokens,
            vocab_size=len(self.keep_tokens)
        )
        self.model = BojoneModel(self.config)
        if hasattr(self.model, "module"):
            self.model.module.load_state_dict(
                torch.load(
                    save_model_path,
                    map_location="cpu"
                ),
                strict=False
            )
        else:
            self.model.load_state_dict(
                torch.load(
                    save_model_path,
                    map_location="cpu"
                ),
                strict=False
            )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        if torch.cuda.device_count() > 1:
            self.model = nn.DataParallel(self.model)
        self.reader = ReadingComprehension(
            start_id=None,
            end_id=self.tokenizer._token_end_id,
            maxlen=gen_max_a_len,
            keep_tokens=self.keep_tokens,
            tokenizer=self.tokenizer,
            config=self.config,
            model=self.model,
            mode='generative'
        )

    def answer(self, question, passage):
        return self.reader.answer(question, [passage])
