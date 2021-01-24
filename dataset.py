# -*- coding: utf-8 -*-
from typing import List
from torch.utils.data import Dataset
from torch.utils.data.sampler import Sampler
import random
import torch
import json
import pickle
from utils import EXPERIMENT_DOMAINS, DONTCARE_TOKEN, NONE_TOKEN, PTR_TOKEN


def get_data(file_name):
    data = []
    domain_counter = {}
    with open(file_name) as f:
        dials = json.load(f)
        for dial_dict in dials:
            # Filtering and counting domains
            for domain in dial_dict["domains"]:
                if domain not in EXPERIMENT_DOMAINS:
                    continue
                if domain not in domain_counter.keys():
                    domain_counter[domain] = 0
                domain_counter[domain] += 1
            dial = []
            # Reading data
            for turn in dial_dict["dialogue"]:
                dial.append({
                    "domain": turn["domain"],  # 'hotel'
                    "turn_id": turn["turn_idx"],  # 0
                    "belief_state": {k: str(v) for k, v in turn["belief_state"].items()},  # {'hotel-area' : 'east', 'hotel-stars': '4'}
                    "transcript": turn["transcript"].strip(),  # '; i need to book a hotel in the east that has 4 stars .'
                    "system_transcript": turn["system_transcript"].strip()
                })
            data.append({
                "id": dial_dict["dialogue_idx"],  # 'PMUL1635.json'
                "domains": dial_dict["domains"],  # ['train', 'hotel']
                "dialogue": dial
            })
    return data, domain_counter


def truncate_seq_pair(tokens_a: List, tokens_b: List=None, max_length=50):
    if tokens_b is None:
        tokens_b = []
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


class DialogDataset(Dataset):
    def __init__(self, data, tokenizer, slot_pair, gate_dict, max_length, bs_max_length, is_train,
                 user_type_id, sys_type_id, belief_type_id, pad_token_id, sep_token_id, cls_token_id):
        self.data = []
        self.tokenizer = tokenizer
        self.pad_id = pad_token_id
        self.cls_token_id = cls_token_id
        self.sep_token_id = sep_token_id
        self.max_length = max_length
        self.user_type_id = user_type_id
        self.sys_type_id = sys_type_id
        self.belief_type_id = belief_type_id
        self.is_train = is_train
        self.gate_dict = gate_dict
        if isinstance(data, str):
            with open(data, mode="rb") as f:
                self.data = pickle.load(f)
        else:
            for each in data:
                history_ids = []
                history_token_type_ids = []
                last_belief_state = {}
                for turn in each["dialogue"]:
                    belief_state = turn["belief_state"]
                    user_tokens = tokenizer.tokenize(turn["transcript"])
                    user_ids = tokenizer.convert_tokens_to_ids(user_tokens)
                    if turn["system_transcript"]:
                        sys_tokens = tokenizer.tokenize(turn["system_transcript"])
                        sys_ids = tokenizer.convert_tokens_to_ids(sys_tokens)
                        text_ids = sys_ids + [sep_token_id] + user_ids
                        text_type_ids = [sys_type_id] * (len(sys_ids) + 1) + [user_type_id] * len(user_ids)
                    else:
                        text_ids = user_ids
                        text_type_ids = [user_type_id] * len(user_ids)
                    history_ids.extend(text_ids + [sep_token_id])
                    history_token_type_ids.extend(text_type_ids + [user_type_id])
                    belief_ids = []
                    for slot, value in last_belief_state.items():
                        tokens = [slot_pair[slot]["domain"], slot_pair[slot]["slot"], "is"] + tokenizer.tokenize(value)
                        belief_ids.extend(tokenizer.convert_tokens_to_ids(tokens) + [sep_token_id])
                    
                    input_ids = history_ids + belief_ids
                    input_type_ids = history_token_type_ids + [belief_type_id] * len(belief_ids)

                    input_ids = [cls_token_id] + input_ids[-(max_length-1):]
                    input_type_ids = [user_type_id] + input_type_ids[-(max_length-1):]

                    targets, labels = [], []
                    target_lens = []
                    for slot in slot_pair:
                        if slot in belief_state.keys():
                            target = tokenizer.tokenize(belief_state[slot])
                            if belief_state[slot] == DONTCARE_TOKEN:
                                label = gate_dict[DONTCARE_TOKEN]
                            elif belief_state[slot] == NONE_TOKEN:
                                label = gate_dict[NONE_TOKEN]
                            else:
                                label = gate_dict[PTR_TOKEN]
                        else:
                            target = tokenizer.tokenize(NONE_TOKEN)
                            label = gate_dict[NONE_TOKEN]
                        target = [slot_pair[slot]["domain"], slot_pair[slot]["slot"]] + target
                        target_ids = tokenizer.convert_tokens_to_ids(target)
                        truncate_seq_pair(target_ids, max_length=bs_max_length-1)
                        targets.append(target_ids+[sep_token_id])
                        target_lens.append(len(target_ids) + 1)
                        labels.append(label)
                    last_belief_state = belief_state
                    self.data.append({
                        "dial_id": each["id"],
                        "dial_domains": each["domains"],
                        "cur_domain": turn["domain"],
                        "turn_id": turn["turn_id"],
                        "input_ids": input_ids,
                        "input_type_ids": input_type_ids,
                        "input_len": len(input_ids),
                        "targets": targets,
                        "target_lens": target_lens,
                        "labels": labels
                    })

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data_item = self.data[idx]
        if self.is_train:
            targets, target_lens = [], []
            for i, label in enumerate(data_item["labels"]):
                if label == self.gate_dict[PTR_TOKEN]:
                    targets.append(data_item["targets"][i])
                    target_lens.append(data_item["target_lens"][i])
            if len(targets) == 0:
                i = random.randint(0, len(data_item["labels"])-1)
                targets.append(data_item["targets"][i])
                target_lens.append(data_item["target_lens"][i])
            data_item["targets"] = targets
            data_item["target_lens"] = target_lens
        return data_item

    def collate_fn(self, data):
        dial_ids = [d["dial_id"] for d in data]
        dial_domains = [d["dial_domains"] for d in data]
        cur_domains = [d["cur_domain"] for d in data]
        turn_ids = [d["turn_id"] for d in data]
        input_ids = [d["input_ids"] for d in data]
        input_type_ids = [d["input_type_ids"] for d in data]
        input_lens = [d["input_len"] for d in data]
        targets = [d["targets"] for d in data]  # [b, slot_len, len]
        target_lens = [d["target_lens"] for d in data]
        labels = [d["labels"] for d in data]

        max_input_len = max(input_lens)
        input_ids = torch.tensor(
            [x + [self.pad_id] * (max_input_len-input_lens[i]) for i, x in enumerate(input_ids)],
            dtype=torch.long
        )
        input_mask = torch.tensor(
            [[1] * input_len + [0] * (max_input_len-input_len) for input_len in input_lens],
            dtype=torch.long
        )
        input_token_type_ids = torch.tensor(
            [x + [0] * (max_input_len-input_lens[i]) for i, x in enumerate(input_type_ids)],
            dtype=torch.long
        )
        max_target_num = max(len(each) for each in targets)
        max_target_len = max([max(each) for each in target_lens])
        target_ids = []
        target_mask = []
        target_seq_mask = []
        for i in range(len(data)):
            each_target_ids = []
            each_target_mask = []
            each_target_seq_mask = []
            for j in range(len(targets[i])):
                each_target_ids.append(targets[i][j] + [self.pad_id] * (max_target_len-target_lens[i][j]))
                each_target_mask.append([1] * target_lens[i][j] + [0] * (max_target_len-target_lens[i][j]))
                each_target_seq_mask.append(1)
            for j in range(max_target_num-len(targets[i])):
                each_target_ids.append([self.pad_id] * max_target_len)
                each_target_mask.append([1] * max_target_len)
                each_target_seq_mask.append(0)
            target_ids.append(each_target_ids)
            target_mask.append(each_target_mask)
            target_seq_mask.append(each_target_seq_mask)
        target_ids = torch.tensor(target_ids, dtype=torch.long)
        target_mask = torch.tensor(target_mask, dtype=torch.long)
        target_seq_mask = torch.tensor(target_seq_mask, dtype=torch.long)
        labels = torch.tensor(labels, dtype=torch.long)
        batch = {
            "dialogue_ids": dial_ids,
            "dialogue_domains": dial_domains,
            "cur_domains": cur_domains,
            "turn_ids": turn_ids,
            "input_ids": input_ids,
            "input_attention_mask": input_mask,
            "input_token_type_ids": input_token_type_ids,
            "target": target_ids,  # [b, slot_len, len]
            "target_attention_mask": target_mask,
            "target_seq_mask": target_seq_mask,
            "labels": labels  # [b, slot_len]
        }
        return batch


class ImbalancedDatasetSampler(Sampler):
    """Samples elements randomly from a given list of indices for imbalanced dataset
    Arguments:
        indices (list, optional): a list of indices
        num_samples (int, optional): number of samples to draw
    """

    def __init__(self, dataset, indices=None, num_samples=None):
                
        # if indices is not provided, 
        # all elements in the dataset will be considered
        self.indices = list(range(len(dataset))) \
            if indices is None else indices
            
        # if num_samples is not provided, 
        # draw `len(indices)` samples in each iteration
        self.num_samples = len(self.indices) \
            if num_samples is None else num_samples
            
        # distribution of classes in the dataset 
        label_to_count = {}
        for idx in self.indices:
            label = self._get_label(dataset, idx)
            if label in label_to_count:
                label_to_count[label] += 1
            else:
                label_to_count[label] = 1
                
        # weight for each sample
        weights = [1.0 / label_to_count[self._get_label(dataset, idx)] for idx in self.indices]
        self.weights = torch.DoubleTensor(weights)

    def _get_label(self, dataset, idx):
        return dataset.turn_domain[idx]
                
    def __iter__(self):
        return (self.indices[i] for i in torch.multinomial(self.weights, self.num_samples, replacement=True))

    def __len__(self):
        return self.num_samples
