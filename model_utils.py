# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import numpy as np


def get_subsequent_mask(seq: torch.LongTensor):
    ''' For masking out the subsequent info. '''
    sz_b, len_s = seq.size()
    subsequent_mask = torch.triu(
        torch.ones((len_s, len_s), device=seq.device, dtype=torch.bool), diagonal=1)
    subsequent_mask = subsequent_mask.unsqueeze(0).expand(sz_b, -1, -1)  # b x ls x ls
    return subsequent_mask


def get_attn_key_pad_mask(k_mask: torch.LongTensor, q_mask: torch.LongTensor):
    ''' For masking out the padding part of key sequence. '''
    # Expand to fit the shape of key query attention matrix.
    head_mask = torch.eq(k_mask, 0)
    head_mask = head_mask.unsqueeze(1).expand(-1, q_mask.size(1), -1)  # b x lq x lk
    return head_mask


class PositionalEmbedding(nn.Module):
    def __init__(self, demb):
        super(PositionalEmbedding, self).__init__()
        inv_freq = 1 / (10000 ** (torch.arange(0.0, demb, 2.0) / demb))
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, pos_seq, bsz=None):
        sinusoid_inp = torch.ger(pos_seq, self.inv_freq)
        pos_emb = torch.cat([sinusoid_inp.sin(), sinusoid_inp.cos()], dim=-1)

        if bsz is not None:
            return pos_emb[None, :, :].expand(bsz, -1, -1)
        else:
            return pos_emb


class PositionWiseFF(nn.Module):
    def __init__(self, d_model, d_inner, dropout, pre_layer_norm=False):
        super(PositionWiseFF, self).__init__()
        self.d_model = d_model
        self.d_inner = d_inner
        self.dropout = dropout
        core_line1 = nn.Linear(d_model, d_inner)
        core_line2 = nn.Linear(d_inner, d_model)
        torch.nn.init.kaiming_normal_(core_line1.weight, nonlinearity='relu')
        nn.init.constant_(core_line1.bias, 0.0)
        torch.nn.init.kaiming_normal_(core_line2.weight, nonlinearity='relu')
        nn.init.constant_(core_line2.bias, 0.0)
        self.CoreNet = nn.Sequential(
            core_line1,
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            core_line2,
            nn.Dropout(dropout)
        )
        self.layer_norm = nn.LayerNorm(d_model)
        self.pre_layer_norm = pre_layer_norm

    def forward(self, inp):
        if self.pre_layer_norm:
            # layer normalization + position wise feed-forward
            core_out = self.CoreNet(self.layer_norm(inp))
            # residual connection
            output = core_out + inp
        else:
            # position wise feed-forward
            core_out = self.CoreNet(inp)
            # residual connection + layer normalization
            output = self.layer_norm(inp + core_out)
        return output


class MultiHeadAttention(nn.Module):
    def __init__(self, n_head, d_model, dropout, pre_layer_norm=False):
        super(MultiHeadAttention, self).__init__()
        self.n_head = n_head
        d_head = d_model // n_head
        self.d_head = d_head
        self.w_qs = nn.Linear(d_model, n_head * d_head)
        self.w_ks = nn.Linear(d_model, n_head * d_head)
        self.w_vs = nn.Linear(d_model, n_head * d_head)
        nn.init.normal_(self.w_qs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_head)))
        nn.init.constant_(self.w_qs.bias, 0.0)
        nn.init.normal_(self.w_ks.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_head)))
        nn.init.constant_(self.w_ks.bias, 0.0)
        nn.init.normal_(self.w_vs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_head)))
        nn.init.constant_(self.w_vs.bias, 0.0)
        self.scale = d_head ** -0.5
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.pre_layer_norm = pre_layer_norm
        self.fc = nn.Linear(n_head * d_head, d_model)
        torch.nn.init.kaiming_normal_(self.fc.weight, nonlinearity='relu')
        torch.nn.init.constant_(self.fc.bias, 0.0)


    def forward(self, q, k, v, mask=None):
        """
        mix original & relative & average multi-head attention
        :param q: B * q_len * d_model
        :param k: B * k_len * d_model
        :param v: B * k_len * d_model
        :param mask: q_len * k_len or B * q_len * k_len
        :return: B * q_len * d_model
        """
        d_head, n_head = self.d_head, self.n_head

        sz_b, q_len, k_len = q.size(0), q.size(1), k.size(1)
        residual = q
        if self.pre_layer_norm:
            q, k, v = self.layer_norm(q), self.layer_norm(k), self.layer_norm(v)
        q = self.w_qs(q).view(sz_b, q_len, n_head, d_head)
        k = self.w_ks(k).view(sz_b, k_len, n_head, d_head)
        v = self.w_vs(v).view(sz_b, k_len, n_head, d_head)

        attn_score = torch.einsum('bqnd,bknd->bqkn', [q, k])
        attn_score.mul_(self.scale)

        if mask is not None and mask.any().item():
            if mask.dim() == 2:
                mask = mask[None, :, :, None]
            elif mask.dim() == 3:
                # mask: B * q_len * k_len
                mask = mask[:, :, :, None]
            attn_score = attn_score.float().masked_fill(mask, -float('inf')).type_as(attn_score)

        # [bsz x qlen x klen x n_head]
        attn_prob = torch.softmax(attn_score, dim=2)
        attn_prob = self.dropout(attn_prob)
        # compute attention vector
        attn_vec = torch.einsum('bqkn,bknd->bqnd', [attn_prob, v])
        attn_vec = attn_vec.contiguous().view(sz_b, q_len, self.n_head * self.d_head)
        # linear projection
        output = self.fc(attn_vec)
        output = self.dropout(output)

        if self.pre_layer_norm:
            # residual connection
            output = residual + output
        else:
            # residual connection + layer normalization
            output = self.layer_norm(residual + output)
        return output, attn_prob


class DecoderLayer(nn.Module):
    def __init__(self, d_model, d_inner, n_head, dropout, pre_layer_norm):
        super(DecoderLayer, self).__init__()
        self.slf_attn = MultiHeadAttention(n_head, d_model, dropout, pre_layer_norm)
        self.enc_attn = MultiHeadAttention(n_head, d_model, dropout, pre_layer_norm)
        self.pos_ffn = PositionWiseFF(d_model, d_inner, dropout, pre_layer_norm)

    def forward(self, dec_inp, enc_out, non_pad_mask=None, slf_attn_mask=None, dec_enc_attn_mask=None):
        dec_out, _ = self.slf_attn(dec_inp, dec_inp, dec_inp, slf_attn_mask)
        dec_out *= non_pad_mask.unsqueeze(-1)
        dec_out, _ = self.enc_attn(dec_out, enc_out, enc_out, dec_enc_attn_mask)
        dec_out *= non_pad_mask.unsqueeze(-1)
        dec_out = self.pos_ffn(dec_out)
        dec_out *= non_pad_mask.unsqueeze(-1)
        return dec_out


class TransformerDecoder(nn.Module):
    def __init__(self, word_emb, n_layer, n_head, d_model, d_inner, dropout, pre_layer_norm, device):
        super(TransformerDecoder, self).__init__()
        self.word_emb = word_emb
        self.pos_emb = PositionalEmbedding(demb=d_model)
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, d_inner, n_head, dropout, pre_layer_norm) for _ in range(n_layer)])
        self.device = device

    def forward(self, target, target_mask, enc_out, enc_mask):
        slf_attn_mask_subseq = get_subsequent_mask(target)
        slf_attn_mask_keypad = get_attn_key_pad_mask(k_mask=target_mask, q_mask=target_mask)
        slf_attn_mask = slf_attn_mask_keypad + slf_attn_mask_subseq
        dec_enc_attn_mask = get_attn_key_pad_mask(k_mask=enc_mask, q_mask=target_mask)

        pos_seq = torch.arange(0.0, target.size(1), 1.0).to(self.device)
        dec_inp = self.word_emb(target) + self.pos_emb(pos_seq, target.size(0))
        for layer in self.layers:
            dec_inp = layer(dec_inp, enc_out, non_pad_mask=target_mask, slf_attn_mask=slf_attn_mask,
                            dec_enc_attn_mask=dec_enc_attn_mask)
        return dec_inp


class EncoderLayer(nn.Module):
    def __init__(self, d_model, d_inner, n_head, dropout, pre_layer_norm):
        super(EncoderLayer, self).__init__()
        self.slf_attn = MultiHeadAttention(n_head, d_model, dropout, pre_layer_norm)
        self.pos_ffn = PositionWiseFF(d_model, d_inner, dropout, pre_layer_norm)

    def forward(self, enc_inp, non_pad_mask=None, slf_attn_mask=None):
        output, _ = self.slf_attn(enc_inp, enc_inp, enc_inp, slf_attn_mask)
        output *= non_pad_mask.unsqueeze(-1)
        output = self.pos_ffn(output)
        output *= non_pad_mask.unsqueeze(-1)
        return output


class TransformerEncoder(nn.Module):
    def __init__(self, word_emb, n_layer, n_head, d_model, d_inner, dropout, pre_layer_norm, device):
        super(TransformerEncoder, self).__init__()
        self.position_emb = PositionalEmbedding(demb=d_model)
        self.token_type_emb = nn.Embedding(2, d_model)
        self.word_emb = word_emb
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, d_inner, n_head, dropout, pre_layer_norm) for _ in range(n_layer)])
        self.device = device

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, return_dict=False):
        src_seq = input_ids.to(self.device)
        slf_attn_mask = get_attn_key_pad_mask(k_mask=src_seq, q_mask=src_seq)
        pos_seq = torch.arange(0.0, src_seq.size(1), 1.0).to(self.device)
        type_seq = self.token_type_emb(token_type_ids)
        enc_inp = self.word_emb(src_seq) + self.position_emb(pos_seq, src_seq.size(0)) + type_seq
        for layer in self.layers:
            enc_inp = layer(enc_inp, non_pad_mask=attention_mask, slf_attn_mask=slf_attn_mask)
        enc_out, cls_out = enc_inp, enc_inp[:, 0]
        return enc_out, cls_out
