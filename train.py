# -*- coding: utf-8 -*-
import json
import os
import torch
import torch.nn as nn
import pickle
import random
import argparse
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel
from transformers import BertTokenizer, get_linear_schedule_with_warmup
from transformers.optimization import AdamW
from tokenizer import Tokenizer
from dataset import DialogDataset, get_data
from model import DialogueModel
from utils import clean_tokens, get_special_ids, NONE_TOKEN, DONTCARE_TOKEN, PTR_TOKEN, CLS_SCALES
from log import create_logger


def get_args():
    parser = argparse.ArgumentParser()
    # Required part
    parser.add_argument("--dataset", default="multiwoz", type=str, choices=["multiwoz", "crosswoz"])
    parser.add_argument("--name", required=True, type=str)
    parser.add_argument("--encoder", default="bert", type=str, choices=["bert", "transformer"])
    parser.add_argument("--local_rank", default=-1, type=int)
    parser.add_argument("--num_workers", default=4, type=int)
    parser.add_argument("--no_cuda", action='store_true')
    parser.add_argument('--seed', type=int, default=42)
    # training part
    parser.add_argument("--num_epochs", default=50, type=int)
    parser.add_argument("--batch_size", default=16, type=int)
    parser.add_argument("--learning_rate", default=2e-4, type=float)
    parser.add_argument("--max_grad_norm", default=5.0, type=float)
    parser.add_argument("--log_step", default=500, type=int)  # bsz: 16 -> step: 3542/epoch
    parser.add_argument("--eval_step", default=2000, type=int)
    parser.add_argument("--warmup_proportion", type=float, default=0.01)
    parser.add_argument("--cls_loss", default=1., type=float)
    parser.add_argument("--gen_loss", default=0.3, type=float)
    # model part
    parser.add_argument("--max_seq_len", default=128, type=int)
    parser.add_argument("--hidden_size", default=256, type=int)
    parser.add_argument("--pre_model", default="./dependences/bert-base-uncased", type=str)
    parser.add_argument("--special_tokens", default="./specail_tokens.txt", type=str)
    parser.add_argument("--n_layer", default=3, type=int)
    parser.add_argument("--n_head", default=4, type=int)
    parser.add_argument("--dropout", default=0.2, type=float)
    parser.add_argument("--pre_layer_norm", default=False, action="store_true")
    # generation part
    parser.add_argument("--max_resp_len", default=7, type=int)
    parser.add_argument("--topk", default=20, type=int)
    parser.add_argument("--topp", default=0.9, type=float)
    args = parser.parse_args()
    args = vars(args)
    return args


def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True


class Trainer(object):
    def __init__(self, model, tokenizer, sepcial_token_ids, slot_map, gate_label, train_dataset, dev_dataset, args, logger, device):
        # normal
        self.args = args
        self.device = device
        self.logger = logger
        self.local_rank = args["local_rank"]
        self.tokenizer = tokenizer
        self.sepcial_token_ids = sepcial_token_ids
        self.slot_map = slot_map
        self.gate_label = gate_label
        # model & data
        if args["local_rank"] == -1:
            self.model = model.to(device)
            self.train_dataloader = DataLoader(
                train_dataset, batch_size=args["batch_size"], shuffle=True,
                num_workers=args["num_workers"], collate_fn=train_dataset.collate_fn)

            self.dev_dataloader = DataLoader(
                dev_dataset, batch_size=args["batch_size"], shuffle=False,
                num_workers=args["num_workers"], collate_fn=dev_dataset.collate_fn)
        else:
            self.model = model.to(device)
            self.model = DistributedDataParallel(model, device_ids=[args["local_rank"]]).to(device)
            train_sampler = DistributedSampler(train_dataset)
            self.train_dataloader = DataLoader(
                train_dataset, sampler=train_sampler, batch_size=args["batch_size"],
                num_workers=args["num_workers"], collate_fn=train_dataset.collate_fn)
            dev_sampler = DistributedSampler(dev_dataset)
            self.dev_dataloader = DataLoader(
                dev_dataset, batch_size=args["batch_size"], num_workers=args["num_workers"],
                sampler=dev_sampler, collate_fn=dev_dataset.collate_fn)

        self.gen_criterion = nn.CrossEntropyLoss(ignore_index=sepcial_token_ids["pad_id"], reduction="none").to(device)
        self.cls_criterion = nn.CrossEntropyLoss(weight=torch.tensor(CLS_SCALES), reduction="mean").to(device)
        if self.args["encoder"] == "bert":
            param_optimizer = list(model.named_parameters())
            optimizer_grouped_parameters = [
                {"params": [p for n, p in param_optimizer if "encoder" in n], "lr": 5e-5},
                {"params": [p for n, p in param_optimizer if "classify" in n], "lr": 5e-5},
                {"params": [p for n, p in param_optimizer if ("encoder" not in n and "classify" not in n)],
                 "lr": args["learning_rate"]}
            ]
            self.optimizer = AdamW(optimizer_grouped_parameters)
        else:
            self.optimizer = AdamW(model.parameters(), lr=args["learning_rate"])
        # total_steps = args["num_epochs"] * len(self.train_dataloader)
        # warmup_steps = int(total_steps * args["warmup_proportion"])
        # self.scheduler = get_linear_schedule_with_warmup(
        #     self.optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)
        self.total_step = 1
        self.best_score = 0
        if self.local_rank in [-1, 0]:
            self.logger.info("steps per epoch: {}".format(len(self.train_dataloader)))

        self.do_gen = True
        if args["gen_loss"] < 0.01:
            self.do_gen = False
            logger.info("*** Only train classify task ***")

    def train(self):
        self.model.train()
        count_gen_loss, count_cls_loss, count_len = 0, 0, 0
        for batch in self.train_dataloader:
            input_ids = batch["input_ids"].to(device=self.device)
            input_attention_mask = batch["input_attention_mask"].to(device=self.device)
            input_token_type_ids = batch["input_token_type_ids"].to(device=self.device)
            target = batch["target"].to(device=self.device)  # [b, slot_len, len]
            target_attention_mask = batch["target_attention_mask"].to(device=self.device)
            target_seq_mask = batch["target_seq_mask"].to(device=self.device)
            labels = batch["labels"].to(device=self.device)  # [b, slot_num]
            all_logits, cls_out = self.model(
                input_ids, input_attention_mask, input_token_type_ids,
                target[:, :, :-1], target_attention_mask[:, :, :-1], self.do_gen)
            if self.do_gen:
                gen_loss = self.gen_criterion(
                    all_logits[:, :, 1:, :].contiguous().view(-1, all_logits.size(-1)),
                    target[:, :, 2:].contiguous().view(-1)
                )  # [b*slot_len*max_len]
                gen_loss = torch.sum(gen_loss.view(*target_seq_mask.size(), -1), dim=-1)
                gen_loss = torch.mean(gen_loss*target_seq_mask)
            else:
                gen_loss = torch.tensor(0, dtype=torch.float)

            cls_loss = self.cls_criterion(
                cls_out.view(-1, cls_out.size(-1)),
                labels.view(-1)
            )  # [b*slot_num]
            # cls_loss = torch.mean(cls_loss.view(*labels_scale.size()) * labels_scale)
            loss = self.args["cls_loss"] * cls_loss + self.args["gen_loss"] * gen_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args["max_grad_norm"])
            self.optimizer.step()
            self.optimizer.zero_grad()
            self.model.zero_grad()
            # self.scheduler.step()
            count_gen_loss += gen_loss.item()
            count_cls_loss += cls_loss.item()
            count_len += 1
            if self.total_step % self.args["log_step"] == 0:
                count_gen_loss /= count_len
                count_cls_loss /= count_len
                self.logger.info("[{}] Loss: gen: {:.4f}, cls: {:.4f}".format(
                    self.total_step, count_gen_loss, count_cls_loss
                ))
                count_gen_loss, count_cls_loss, count_len = 0, 0, 0

            if self.total_step % self.args["eval_step"] == 0:
                self.model.eval()
                if self.do_gen:
                    _, slot_acc, _, eval_records = self._eval_test()
                    if slot_acc > self.best_score and self.local_rank in [-1, 0]:
                        self.logger.info(
                            "Slot Acc {:.2f}% -> {:.2f}%, Update Best model".format(slot_acc*100, self.best_score*100))
                        self.save_model()
                        self.save_records(eval_records)
                        self.best_score = slot_acc
                else:
                    gate_acc, gate_records = self._eval_classify()
                    if gate_acc > self.best_score and self.local_rank in [-1, 0]:
                        self.logger.info(
                            "Gate Acc {:.2f}% -> {:.2f}%, Update Best model".format(gate_acc*100, self.best_score*100))
                        self.save_model()
                        with open(os.path.join(self.args["output_dir"], "gate_records.json"), mode="w") as f:
                            json.dump(gate_records, f, ensure_ascii=False)
                        self.best_score = gate_acc
                self.model.train()
            self.total_step += 1

        self.model.eval()
        if self.do_gen:
            _, slot_acc, _, eval_records = self._eval_test()
            if slot_acc > self.best_score and self.local_rank in [-1, 0]:
                self.logger.info(
                    "Slot Acc {:.2f}% -> {:.2f}%, Update Best model".format(slot_acc*100, self.best_score*100))
                self.save_model()
                self.save_records(eval_records)
                self.best_score = slot_acc
        else:
            gate_acc, gate_records = self._eval_classify()
            if gate_acc > self.best_score and self.local_rank in [-1, 0]:
                self.logger.info(
                    "Gate Acc {:.2f}% -> {:.2f}%, Update Best model".format(gate_acc*100, self.best_score*100))
                self.save_model()
                with open(os.path.join(self.args["output_dir"], "gate_records.json"), mode="w") as f:
                    json.dump(gate_records, f, ensure_ascii=False)
                self.best_score = gate_acc
        self.model.train()

    def save_records(self, records):
        new_records = {}
        for dial_name in records:
            turns = [[k, v] for k, v in records[dial_name].items()]
            turns.sort(key=lambda x: x[0])
            new_records[dial_name] = []
            for i in range(len(turns)):
                new_records[dial_name].append(turns[i][1])
        with open(os.path.join(self.args["output_dir"], "eval_records.json"), mode="w") as f:
            json.dump(new_records, f, indent=2, ensure_ascii=False)

    def _eval_test(self):
        back_remoeved_ids = [self.sepcial_token_ids["pad_id"], self.sepcial_token_ids["eos_id"]]
        slot_correct, label_correct, joint_correct = 0, 0, 0
        extra_slot_correct, extra_slot_count = 0, 0
        tn, fn = 0, 0
        turn_count, slot_count = 0, 0
        eval_records = {}
        slot_num = len(self.slot_map)
        with torch.no_grad():
            all_domain_slot = [[v["domain"], v["slot"]] for v in self.slot_map.values()]
            all_domain_ids = [self.tokenizer.convert_tokens_to_ids(domain_slot) for domain_slot in all_domain_slot]
            slot_ids = torch.tensor(all_domain_ids, dtype=torch.long, device=self.device)
            for batch in self.dev_dataloader:
                input_ids = batch["input_ids"].to(device=self.device)
                input_attention_mask = batch["input_attention_mask"].to(device=self.device)
                input_token_type_ids = batch["input_token_type_ids"].to(device=self.device)
                target_labels = batch["labels"].to(device=self.device)
                model = self.model.module if hasattr(self.model, 'module') else self.model
                all_results, cls_out = model.generate(
                    input_ids, input_attention_mask, input_token_type_ids, slot_ids, self.args["topk"],
                    self.args["topp"], self.gate_label[NONE_TOKEN], self.gate_label[DONTCARE_TOKEN],
                    self.sepcial_token_ids["none_id"], self.sepcial_token_ids["dontcare_id"],
                    self.sepcial_token_ids["eos_id"], self.args["max_resp_len"], target_labels=target_labels
                )
                bsz, slot_num = cls_out.size()
                cls_out = cls_out.cpu().detach().tolist() # [b, slot_num]
                labels = batch["labels"].tolist()  # [b, slot_num]
                all_results = all_results.cpu().detach().tolist()  # [b, slot_num, max_len]
                targets = batch["target"].tolist()  # [b, slot_num, max_len]
                for i in range(bsz):
                    dial_id, turn_id = batch["dialogue_ids"][i], batch["turn_ids"][i]
                    if dial_id not in eval_records:
                        eval_records[dial_id] = {}
                    if turn_id not in eval_records[dial_id]:
                        eval_records[dial_id][turn_id] = {}
                    joint_done = True
                    for j in range(slot_num):
                        domain_slot = "-".join(all_domain_slot[j])
                        result = clean_tokens(all_results[i][j], back_remoeved_ids=back_remoeved_ids)
                        target = targets[i][j]
                        slot, target = target[:2], target[2:]
                        target = clean_tokens(target, back_remoeved_ids=back_remoeved_ids)
                        assert slot == all_domain_ids[j], "{}, {}".format(slot, all_domain_ids[j])
                        eval_records[dial_id][turn_id][domain_slot]= {
                            "pred": " ".join(self.tokenizer.convert_ids_to_tokens(result)),
                            "ref": " ".join(self.tokenizer.convert_ids_to_tokens(target)),
                            "pred_gate": cls_out[i][j],
                            "ref_gate": labels[i][j]
                        }
                        if cls_out[i][j] == labels[i][j]:
                            label_correct += 1
                            if cls_out[i][j] == self.gate_label[NONE_TOKEN]:
                                tn += 1
                        else:
                            if cls_out[i][j] == self.gate_label[NONE_TOKEN]:
                                fn += 1
                            joint_done = False

                        if labels[i][j] == self.gate_label[PTR_TOKEN]:
                            if set(result) == set(target):
                                slot_correct += 1
                            else:
                                joint_done = False
                            slot_count += 1
                        else:
                            if cls_out[i][j] == labels[i][j]:
                                extra_slot_correct += 1
                            extra_slot_count += 1
                    if joint_done:
                        joint_correct += 1
                    turn_count += 1
            gate_acc = label_correct / (turn_count * slot_num)
            slot_acc = (slot_correct + extra_slot_correct) / (slot_count + extra_slot_count)
            pure_slot_acc = slot_correct / slot_count
            joint_acc = joint_correct / turn_count
            self.logger.info(
                "[Step {}] Accuracy: Joint {:.2f}%, Slot: {:.2f}%, Pure Slot: {:.2f}%, Gate: {:.2f}%".format(
                    self.total_step, joint_acc*100, slot_acc*100, pure_slot_acc*100, gate_acc*100)
            )
            self.logger.info(
                "[Slot Gate]: Total: {}, Ture None: {}, False None: {}".format(turn_count * slot_num, tn, fn)
            )
            rand_dial = random.choice(list(eval_records.keys()))
            rand_turn = random.choice(list(eval_records[rand_dial].keys()))
            log_str = "---- Dial: {}, Turn: {} ----\n".format(rand_dial, rand_turn)
            for _ in range(2):
                rand_ds = random.choice(list(eval_records[rand_dial][rand_turn].keys()))
                log_str += "Domain-Slot: {}\n".format(rand_ds)
                log_str += "\n".join(
                    ["{}: {}".format(k, v) for k, v in eval_records[rand_dial][rand_turn][rand_ds].items()]
                )
                log_str += "\n"
            self.logger.info(log_str)
        return joint_acc, slot_acc, gate_acc, eval_records

    def _eval_classify(self):
        joint_gate_correct, gate_correct, total_num = 0, 0, 0
        slot_num = len(self.slot_map)
        records = []
        with torch.no_grad():
            for batch in self.dev_dataloader:
                input_ids = batch["input_ids"].to(device=self.device)
                input_attention_mask = batch["input_attention_mask"].to(device=self.device)
                input_token_type_ids = batch["input_token_type_ids"].to(device=self.device)
                target = batch["target"].to(device=self.device)  # [b, slot_len, len]
                labels = batch["labels"]
                _, cls_out = self.model(
                    input_ids, input_attention_mask, input_token_type_ids,
                    target, None, False)
                preds = torch.argmax(cls_out, dim=-1).detach().cpu()  # [b, slot_num]
                for i in range(len(batch["dialogue_ids"])):
                    records.append({
                        "dialogue_id": batch["dialogue_ids"][i],
                        "turn_id": batch["turn_ids"][i],
                        "preds": preds[i].tolist(),
                        "labels": labels[i].tolist()
                    })
                    eq_num = torch.eq(preds[i], labels[i]).sum().item()
                    gate_correct += eq_num
                    if eq_num == slot_num:
                        joint_gate_correct += 1
                    total_num += 1
        gate_acc = gate_correct / (total_num * slot_num)
        joint_gate_acc = joint_gate_correct / total_num
        self.logger.info("[Step {}] Gate Acc: {:.2f}% Joint Gate Acc: {:.2f}%".format(
            self.total_step, gate_acc*100, joint_gate_acc*100))
        return gate_acc, records

    def save_model(self):
        model_path = os.path.join(self.args["output_dir"], "model.bin")
        model_to_save = self.model.module if hasattr(self.model, 'module') else self.model
        torch.save(model_to_save.state_dict(), model_path)


def main():
    args = get_args()
    setup_seed(args["seed"])
    output_dir = os.path.join("output", args["name"])
    data_dir = os.path.join("data", args["dataset"])
    args["output_dir"] = output_dir
    args["data_dir"] = data_dir
    while not os.path.exists(output_dir):
        if args["local_rank"] in [-1, 0]:
            os.mkdir(output_dir)
    logger = create_logger(os.path.join(output_dir, 'train.log'), local_rank=args["local_rank"])
    if args["local_rank"] in [-1, 0]:
        logger.info(args)
        with open(os.path.join(output_dir, "args.json"), mode="w") as f:
            json.dump(args, f)
    # code for distributed training
    if args["local_rank"] != -1:
        device = torch.device("cuda:{}".format(args["local_rank"]))
        torch.distributed.init_process_group(backend='nccl', init_method='env://', world_size=4)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() and not args["no_cuda"] else "cpu")
    
    # slot & gate
    with open(os.path.join(data_dir, "slot_map.json")) as f:
        slot_map = json.load(f)
    with open(os.path.join(data_dir, "gate_label.txt")) as f:
        gate_label = {line.strip(): i for i, line in enumerate(f)}
    
    if args["encoder"] == "bert":
        if len(args["special_tokens"]) > 0 and os.path.exists(args["special_tokens"]):
            with open(args["special_tokens"]) as f:
                special_tokens = f.read().strip().split("\n")
            tokenizer = BertTokenizer.from_pretrained(args["pre_model"], additional_special_tokens=special_tokens)
        else:
            tokenizer = BertTokenizer.from_pretrained(args["pre_model"])
        sp_ids = get_special_ids(tokenizer)
        model = DialogueModel(
            args["pre_model"], 0, 0, len(slot_map), len(gate_label), args["n_layer"],
            args["n_head"], args["dropout"], args["pre_layer_norm"], device, sp_ids["pad_id"])
    else:
        tokenizer = Tokenizer(os.path.join(data_dir, "vocab.txt"), True)
        sp_ids = get_special_ids(tokenizer)
        model = DialogueModel(
            args["encoder"], len(tokenizer), args["hidden_size"], len(slot_map), len(gate_label), 
            args["n_layer"], args["n_head"], args["dropout"], args["pre_layer_norm"], device, sp_ids["pad_id"])

    # train_dataset
    train_pkl = os.path.join(data_dir, "train_dials_{}.pkl".format(len(tokenizer)))
    if os.path.exists(train_pkl):
        train_data = train_pkl
        logger.info("load training cache from {}".format(train_pkl))
    else:
        train_data, domain_counter = get_data(os.path.join(data_dir, "train_dials.json"))
        logger.info("Traning domain_counter: {}".format(domain_counter))
    train_dataset = DialogDataset(
        train_data, tokenizer, slot_map, gate_label, args["max_seq_len"], args["max_resp_len"], True,
        sp_ids["user_type_id"], sp_ids["sys_type_id"], sp_ids["belief_type_id"],
        sp_ids["pad_id"], sp_ids["eos_id"], sp_ids["cls_id"], sp_ids["belief_sep_id"]
    )
    if not os.path.exists(train_pkl) and args["local_rank"] in [-1, 0]:
        with open(train_pkl, mode="wb") as f:
            pickle.dump(train_dataset.data, f)
        logger.info("save training cache to {}".format(train_pkl))
    # dev_dataset
    dev_pkl = os.path.join(data_dir, "dev_dials_{}.pkl".format(len(tokenizer)))
    if os.path.exists(dev_pkl):
        dev_data = dev_pkl
        logger.info("load dev cache from {}".format(dev_pkl))
    else:
        dev_data, domain_counter = get_data(os.path.join(data_dir, "dev_dials.json"))
        logger.info("Eval domain_counter: {}".format(domain_counter))
    dev_dataset = DialogDataset(
        dev_data, tokenizer, slot_map, gate_label, args["max_seq_len"], args["max_resp_len"], False,
        sp_ids["user_type_id"], sp_ids["sys_type_id"], sp_ids["belief_type_id"],
        sp_ids["pad_id"], sp_ids["eos_id"], sp_ids["cls_id"], sp_ids["belief_sep_id"]
    )
    if not os.path.exists(dev_pkl) and args["local_rank"] in [-1, 0]:
        with open(dev_pkl, mode="wb") as f:
            pickle.dump(dev_dataset.data, f)
        logger.info("save dev cache to {}".format(dev_pkl))

    trainer = Trainer(model, tokenizer, sp_ids, slot_map, gate_label, train_dataset, dev_dataset, args, logger, device)
    if args["local_rank"] in [-1, 0]:
        logger.info("Start training")

    for epoch in range(1, args["num_epochs"]):
        logger.info("Epoch {} start, Cur step: {}".format(epoch, trainer.total_step))
        trainer.train()


if __name__ == "__main__":
    main()
