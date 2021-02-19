# -*- coding: utf-8 -*-
from collections import Counter
import numpy as np

EXPERIMENT_DOMAINS = ["hotel", "train", "restaurant", "attraction", "taxi"]
CLS_TOKEN = "[CLS]"
SEP_TOKEN = "[SEP]"
PAD_TOEKN = "[PAD]"
UNK_TOKEN = "[UNK]"
BELIEF_SEP_TOKEN = '[B_SEP]'
NONE_TOKEN = "none"
DONTCARE_TOKEN = "dontcare"
PTR_TOKEN = "ptr"
CLS_SCALES = [1.0, 1.0, 1.0]


def get_special_ids(tokenizer):
    cls_id, eos_id, pad_id, none_id, dontcare_id, belief_sep_id = tokenizer.convert_tokens_to_ids(
        [CLS_TOKEN, SEP_TOKEN, PAD_TOEKN, NONE_TOKEN, DONTCARE_TOKEN, BELIEF_SEP_TOKEN])
    return {
        "cls_id": cls_id,
        "eos_id": eos_id,
        "pad_id": pad_id,
        "none_id": none_id,
        "dontcare_id": dontcare_id,
        "belief_sep_id": belief_sep_id,
        "user_type_id": 0,
        "sys_type_id": 1,
        "belief_type_id": 0
    }


def clean_tokens(tokens, front_removed_ids=[], back_remoeved_ids=[]):
    while len(tokens) > 0 and tokens[0] in front_removed_ids:
        tokens = tokens[1:]
    while len(tokens) > 0 and tokens[-1] in back_remoeved_ids:
        tokens = tokens[:-1]
    return tokens


def f1_score(predictions, targets, average=True):
    def f1_score_items(pred_items, gold_items):
        common = Counter(gold_items) & Counter(pred_items)
        num_same = sum(common.values())

        if num_same == 0:
            return 0

        precision = num_same / len(pred_items)
        recall = num_same / len(gold_items)
        f1 = (2 * precision * recall) / (precision + recall)
        return f1
    scores = [f1_score_items(p, t) for p, t in zip(predictions, targets)]
    if average:
        return sum(scores) / len(scores)

    return scores


def wer(r, h):
    """
    This is a function that calculate the word error rate in ASR.
    You can use it like this: wer("what is it".split(), "what is".split()) 
    """
    #build the matrix
    d = np.zeros((len(r)+1)*(len(h)+1), dtype=np.uint8).reshape((len(r)+1, len(h)+1))
    for i in range(len(r)+1):
        for j in range(len(h)+1):
            if i == 0: d[0][j] = j
            elif j == 0: d[i][0] = i
    for i in range(1,len(r)+1):
        for j in range(1, len(h)+1):
            if r[i-1] == h[j-1]:
                d[i][j] = d[i-1][j-1]
            else:
                substitute = d[i-1][j-1] + 1
                insert = d[i][j-1] + 1
                delete = d[i-1][j] + 1
                d[i][j] = min(substitute, insert, delete)
    result = float(d[len(r)][len(h)]) / len(r) * 100
    # result = str("%.2f" % result) + "%"
    return result
