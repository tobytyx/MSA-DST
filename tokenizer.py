# -*- coding: utf-8 -*-
from typing import List
import regex as re
import json
from utils import UNK_TOKEN, CLS_TOKEN, SEP_TOKEN, PAD_TOEKN


class Tokenizer(object):
    def __init__(self, vocab_file, lower_case=True):
        super(Tokenizer, self).__init__()
        self.lower_case = lower_case
        self.ivocab = {}
        self.vocab = {}
        with open(vocab_file) as f:
            for i, line in enumerate(f):
                line = line.strip()
                self.vocab[line] = i
                self.ivocab[i] = line
        self.unk_token_id = self.vocab[UNK_TOKEN]
        self.vocab_len = len(self.vocab)

    def tokenize(self, sent):
        if self.lower_case:
            return re.split(r"\s+", sent.lower())
        else:
            return re.split(r"\s+", sent)
    
    def __len__(self):
        return self.vocab_len

    def get_word_id(self, w):
        return self.vocab.get(w, self.unk_token_id)

    def get_id_word(self, i):
        return self.ivocab.get(i, UNK_TOKEN)

    def convert_tokens_to_ids(self, sent):
        return [self.get_word_id(w) for w in sent]

    def convert_ids_to_tokens(self, word_ids):
        return  [self.get_id_word(wid) for wid in word_ids]


def create_vocab(data_files: List[str], output_file, special_token_file="", min_times=1):
    specials = [PAD_TOEKN, CLS_TOKEN, SEP_TOKEN, UNK_TOKEN]
    if special_token_file:
        with open(special_token_file) as f:
            for line in f:
                line = line.strip()
                specials.append(line)

    vocab_count = {}
    for data_file in data_files:
        with open(data_file) as f:
            dialogues = json.load(f)
        for dial in dialogues:
            for turn in dial["dialogue"]:
                if "system_transcript" in turn and len(turn["system_transcript"]) > 0:
                    for token in turn["system_transcript"].strip().split(" "):
                        if token not in vocab_count:
                            vocab_count[token] = 1
                        else:
                            vocab_count[token] += 1
                if "transcript" in turn and len(turn["transcript"]) > 0:
                    for token in turn["transcript"].strip().split(" "):
                        if token not in vocab_count:
                            vocab_count[token] = 1
                        else:
                            vocab_count[token] += 1
                if "belief_state" in turn and len(turn["belief_state"]) > 0:
                    for v in turn["belief_state"].values():
                        for token in v.strip().split(" "):
                            if token not in vocab_count:
                                vocab_count[token] = 1
                            else:
                                vocab_count[token] += 1
    vocab = specials + [k for k, v in vocab_count.items() if v >= min_times]
    print("total vocab: {}".format(len(vocab)))
    with open(output_file, mode="w") as f:
        for k in vocab:
            f.write(k+"\n")


if __name__ == "__main__":
    create_vocab(
        ["./data/multiwoz/train_dials.json", "./data/multiwoz/dev_dials.json", "./data/multiwoz/test_dials.json"],
        "./data/multiwoz/vocab.txt",
        "special_tokens.txt",
        2
    )
