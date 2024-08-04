import pandas as pd
import ast
import re

def get_clean_phrase_tokenized(text):
    """
    reads phrase tokenized as string and concatenates the tokens
    """
    print(text)
    pattern = r'(\d+:=)'
    parts = re.split(pattern, text)
    tokens = []
    for i in range(1, len(parts), 2):
        tokens.append(parts[i] + parts[i+1].strip())

    print(tokens)
    strip_tokens = [tok.split(":=")[1] for tok in tokens]
    max_num_tokens = int(tokens[-1].split(":=")[0]) + 1
    return strip_tokens, max_num_tokens

def construct_spans(statement_span,max_num_tokens):
    """
    Input: [[2,4],[6,7,8]],10
    Output: [0,0,1,0,1,0,2,2,2,0,0]
    """
    spans = [0] * max_num_tokens
    c = 1
    if statement_span == []:
        return spans
    else:
        print(statement_span)
        for span in statement_span:
            for i in span:
                spans[i] = c
            c += 1
        return spans

if __name__ == "__main__":
    data = pd.read_csv("../data/train_trial_test.csv").fillna("")
    phrase_tokenized = data["phrase_tokenized"].tolist()
    statement_spans = data["statement_spans"].tolist()
    sent_ids = data["sent-id"].tolist()
    tokens = []
    max_num_tokens = []

    for item in phrase_tokenized:
        tok, max_num_tok = get_clean_phrase_tokenized(item)
        tokens.append(tok)
        max_num_tokens.append(max_num_tok)

    statement_spans = []
    for item in data["statement_spans"].tolist():
        if item:
            statement_spans.append(ast.literal_eval(item))
        else:
            statement_spans.append([])
    data_dict = []
    for i in range(len(tokens)):
        data_dict.append({"sent-id": sent_ids[i], "tokens": tokens[i], "labels": construct_spans(statement_spans[i],max_num_tokens[i])})

    df = pd.DataFrame(data_dict)
    df.to_csv("../data/subtask2_trial_train_test.csv",index=False)
