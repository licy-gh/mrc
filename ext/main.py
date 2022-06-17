from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline

if __name__ == "__main__":
    p = "据报道,这条拟建的运河长达2300英里(约合3680公里),如果运河最终建成,将成为世界上最长的人工运河.而现在世界上最长的人工运河是中国的京杭大运河,长1120英里(1794公里)."

    q = "世界上最长的一条人工河是我国的哪一条河？"

    # "京杭大运河",70

    model = AutoModelForQuestionAnswering.from_pretrained('./Output/model')

    tokenizer = AutoTokenizer.from_pretrained('./Output/model')

    QA = pipeline('question-answering', model=model, tokenizer=tokenizer)

    QA_input = {'question': q, 'context': p}

    ans = QA(QA_input)

    a = ans['answer']

    ast = ans['start']

    aed = ans['end']

    print('passage:{}, question:{}, answer:{}, answer_start:{}, answer_end:{}'.format(p, q, a, ast, aed))
