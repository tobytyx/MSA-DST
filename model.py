# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from transformers import BertModel
from transformers.generation_utils import top_k_top_p_filtering
from model_utils import TransformerDecoder, TransformerEncoder


class DialogueModel(nn.Module):
    def __init__(self, encoder_name, vocab_size, hidden_size, slot_num, num_labels, n_layer, n_head, dropout, pre_layer_norm, device, pad_id):
        super().__init__()
        if encoder_name == "transformer":
            word_embedings = nn.Embedding(vocab_size, hidden_size, padding_idx=pad_id)
            d_inner = hidden_size * 4
            self.encoder = TransformerEncoder(word_embedings, n_layer, n_head, hidden_size, d_inner, dropout, pre_layer_norm, device)
        else:
            self.encoder = BertModel.from_pretrained(encoder_name)
            word_embedings = self.encoder.get_input_embeddings()
            hidden_size = self.encoder.config.hidden_size
            d_inner = hidden_size * 4
        self.d_model = hidden_size
        self.slot_num = slot_num
        self.classify = nn.Linear(hidden_size, slot_num * num_labels)
        self.decoder = TransformerDecoder(word_embedings, n_layer, n_head, hidden_size, d_inner, dropout, pre_layer_norm, device)
        self.lm_head = nn.Linear(hidden_size, word_embedings.num_embeddings, bias=False)
        self.lm_head.weight = word_embedings.weight
        self.dropout = nn.Dropout(dropout)
        self.device = device

    def forward(self, input_ids, input_attention_mask, input_token_type_ids, target, target_attention_mask, do_gen=True):
        enc_out, cls_out, *_ = self.encoder(input_ids, attention_mask=input_attention_mask, token_type_ids=input_token_type_ids, return_dict=False)
        bsz, target_slot_len, _ = target.size()
        cls_out = self.classify(cls_out)
        cls_out = cls_out.view(bsz, self.slot_num, -1)  # [b, slot_num, num_labels]
        if do_gen:
            all_logits = []
            for i in range(target_slot_len):
                slot_target = target[:, i, :]
                slot_target_mask = target_attention_mask[:, i, :]
                dec_out = self.decoder(slot_target, slot_target_mask, enc_out, input_attention_mask)
                logits = self.lm_head(dec_out)
                all_logits.append(logits)
            all_logits = torch.stack(all_logits, dim=0).transpose(1, 0)  # [b, t_slot_len, len]
        else:
            all_logits = None
        return all_logits, cls_out

    def generate(self, input_ids, input_attention_mask, input_token_type_ids, slot_ids, top_k, top_p,
                 none_label, dontcare_label, none_id, dontcare_id, eos_id, max_resp_len, target_labels=None):
        """ slot_ids: [slot_num, 2]  """
        batch_size = input_ids.size(0)
        enc_out, cls_out, *_ = self.encoder(input_ids, attention_mask=input_attention_mask, token_type_ids=input_token_type_ids, return_dict=False)
        cls_out = self.classify(cls_out)
        cls_out = cls_out.view(batch_size, self.slot_num, -1)
        cls_out = torch.argmax(cls_out, dim=-1)  # [b, slot_num]
        all_result = torch.ones(batch_size, self.slot_num, max_resp_len, dtype=cls_out.dtype, device=self.device) * eos_id  # [b, slot_num, max_len]
        if target_labels is None:
            target_labels = cls_out
        for i in range(self.slot_num):
            cur_len = 0
            is_done = torch.zeros(batch_size, dtype=torch.bool, device=self.device)
            is_none = torch.eq(target_labels[:, i], none_label).to(device=self.device)
            is_dontcare = torch.eq(target_labels[:, i], dontcare_label).to(device=self.device)
            is_done = is_done + is_none + is_dontcare
            slot_target = slot_ids[i].unsqueeze(0).expand(batch_size, -1)  # [b, 2]
            slot_target_mask = torch.ones(*slot_target.size(), dtype=torch.long, device=self.device)
            while cur_len < max_resp_len:
                dec_out = self.decoder(slot_target, slot_target_mask, enc_out, input_attention_mask)
                logits = self.lm_head(dec_out[:, -1, :])
                logits = top_k_top_p_filtering(logits, top_k=top_k, top_p=top_p)
                probs = torch.softmax(logits, dim=-1)
                next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)  # [batch_size]
                is_done = is_done + torch.eq(next_tokens, eos_id)
                next_tokens[is_done] = eos_id
                all_result[:, i, cur_len] = next_tokens
                cur_len = cur_len + 1
                if all(is_done):
                    break
            all_result[is_none, i, 0] = none_id
            all_result[is_dontcare, i, 0] = dontcare_id
        return all_result, cls_out
