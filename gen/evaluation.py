import os.path
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge import Rouge
from config import *
from rc import GenAnswer
from utils import divide_data

import jieba

rouge = Rouge()
smooth = SmoothingFunction().method1


def eval_valid_data(tmp_valid_data, gen, enum=-1, show_case=-1, full_test=False):
    eval_log = os.path.join(log_dir, f"full_test.json") if full_test else os.path.join(log_dir,
                                                                                       f"P_s2s_pred_answer_{enum}.json")
    eval_f = open(eval_log, mode='w', encoding='utf-8')
    output_dict = []
    total = 0
    rouge_1, rouge_2, rouge_l, bleu = 0, 0, 0, 0
    gen.model.eval()
    for i, d in enumerate(tqdm(tmp_valid_data)):
        qid = d["id"]
        q_text = d['question']
        p_text = d['context']
        fact_a = 'noans' if d['is_impossible'] else d['answers'][0]['text']
        pred_a = gen.answer(q_text, p_text)

        if (d['is_impossible']) and pred_a == 'noans':
            curr_rouge_1, curr_rouge_2, curr_rouge_l, curr_bleu = 1, 1, 1, 1
        else:
            fact_a_c = jieba.cut(fact_a)
            fact_a_c = ' '.join(fact_a_c)
            pred_a_c = jieba.cut(pred_a)
            pred_a_c = ' '.join(pred_a_c)

            try:
                scores = rouge.get_scores(hyps=pred_a_c, refs=fact_a_c)
                curr_rouge_1 = scores[0]['rouge-1']['f']
                curr_rouge_2 = scores[0]['rouge-2']['f']
                curr_rouge_l = scores[0]['rouge-l']['f']
                curr_bleu = sentence_bleu(
                    references=[fact_a_c],
                    hypothesis=pred_a_c,
                    smoothing_function=smooth
                )
            except ValueError as err:
                curr_rouge_1, curr_rouge_2, curr_rouge_l, curr_bleu = 0, 0, 0, 0
                logger.error(
                    f'[{err}] qid: {qid} question: {q_text} fact_a: {fact_a} pred_a: {pred_a}')

        if i < show_case:
            logger.info(f"qid: {qid} q_text: {q_text} a_text: {pred_a}")

        output_dict.append({
            "qid": str(qid),
            "question": q_text,
            "pred_ans": pred_a,
            "fact_ans": fact_a,
            "ROUGE-1": curr_rouge_1,
            "ROUGE-2": curr_rouge_2,
            "ROUGE-L": curr_rouge_l,
            "BLEU": curr_bleu
        })

        total += 1
        rouge_1 += curr_rouge_1
        rouge_2 += curr_rouge_2
        rouge_l += curr_rouge_l
        bleu += curr_bleu

    rouge_1 /= total
    rouge_2 /= total
    rouge_l /= total
    bleu /= total

    enum_str = f'Epoch {enum}' if enum >= 0 else 'FULL'
    logger.info(
        f"EVAL {enum_str}: ROUGE_1={rouge_1} ROUGE_2={rouge_2} ROUGE_L={rouge_l} BLEU={bleu}")

    output_dict.append({
        "qid": "[THE_END]",
        "question": "[THE_END]",
        "pred_ans": "[THE_END]",
        "fact_ans": "[THE_END]",
        "ROUGE-1": rouge_1,
        "ROUGE-2": rouge_2,
        "ROUGE-L": rouge_l,
        "BLEU": bleu
    })
    json.dump(obj=output_dict, fp=eval_f, ensure_ascii=False, indent=4)

    return rouge_1, rouge_2, rouge_l, bleu


if __name__ == "__main__":
    gen_ans = GenAnswer()
    data_f = open(mrc_dataset_path, mode='r', encoding='utf-8')
    # 标注数据
    all_data = json.load(data_f)
    # 保存一个随机序（供划分valid用）
    train_data, valid_data = divide_data(all_data)

    logger.info("begin final evaluation ...")
    eval_valid_data(
        tmp_valid_data=valid_data,
        gen=gen_ans,
        full_test=True
    )
    logger.info("end final evaluation ...")
