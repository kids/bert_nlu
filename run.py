# -*- coding: utf-8 -*-

import torch
import os
import random
import logging

from transformers import BertTokenizer, BertModel, BertConfig
from torch.utils.data import TensorDataset, DataLoader, RandomSampler

from prepare import mk_tgts, mk_dataset
from model import DmBERT
from train import train_model


def parse_slots(slot_logits, s, tok, slot_logit_thresh=1.9):
    token_len = len(slot_logits[0])
    max_v, max_i = torch.max(slot_logits[0], 1)
    # print(token_len,max_v[:token_len],max_i[:token_len])
    # max_i[max_v<slot_logit_thresh]=0
    slot_bid = (max_i[:token_len] % 2 == 1).nonzero().reshape(-1).tolist()
    slot_bid.append(token_len)
    # print(slot_bid)
    ret = []
    for i in range(len(slot_bid) - 1):
        slot_name = slot_label_list[max_i[slot_bid[i]]]
        cret = [slot_name, slot_bid[i]]
        for j in range(slot_bid[i] + 1, slot_bid[i + 1]):
            if (max_i[j] - max_i[slot_bid[i]]) != 1:
                cret.append(j)
                ret.append(cret)
                break
        if len(cret) == 2:
            cret.append(slot_bid[i + 1])
            ret.append(cret)

    l = tok[0][1:-1]
    rs = []
    for i in range(len(l)):
        if l[i] == 100:
            n = s.find(tokenizer.ids_to_tokens[int(l[i + 1])])
        else:
            n = 1
        rs.append(s[:n])
        s = s[n:]
    rs = ['|'] + rs + ['|']

    nret = []
    for cret in ret:
        cur_slot_score = torch.mean(max_v[cret[1]:cret[2]]
                                    ).detach().numpy().tolist()
        cur_slot_text = ''.join(rs[cret[1]:cret[2]])
        nret.append([cret[0], cur_slot_text, cur_slot_score])

    return nret


def main(training):
    cur_path = os.path.abspath(os.path.dirname(__file__))
    mdl_path = os.path.join(cur_path, 'mdl/')
    raw_data = os.path.join(cur_path, 'data/dmslot.txt')
    data_set = os.path.join(cur_path, 'data/dmslot')

    tokenizer = BertTokenizer.from_pretrained(mdl_path)
    cfg = BertConfig.from_pretrained(mdl_path)
    max_length = cfg.max_position_embeddings
    padding_id = cfg.pad_token_id

    domain_map, domain_label_list, slot_map, slot_label_list = mk_tgts()
    # dataset = mk_dataset(raw_data, data_set, tokenizer)

    # if os.path.exists(mdl_path):
    #     model = DmBERT.from_pretrained(
    #         mdl_path,
    #         intent_label_lst=domain_label_list,
    #         slot_label_lst=slot_label_list)

    model = DmBERT(cfg, domain_label_list, slot_label_list)
    # train_model(model, dataset, os.path.join(mdl_path, 'sess'))


    model = torch.quantization.quantize_dynamic(model,dtype=torch.qint8)
    model = torch.load(os.path.join(mdl_path, 'pytorch_model.bin'))

    return tokenizer, model


if __name__ == '__main__':
    domain_map, domain_label_list, slot_map, slot_label_list = mk_tgts()
    tokenizer, model = main(training=False)
    # test
    for s in [
        '给我来首好听的歌',
        '经过三里河去五月天家',
        '座椅降低2度',
        '你是不是傻',
        '太阳半径是多少',
        '设个明天早晨10点的闹钟',
        '去最近的川菜馆',
        '播放momo的新剧',
        '打开qq音乐放收藏列表',
        '平江路堵不堵',
        '生活的快乐与否，完全决定于个人对人、事、物的看法如何'
    ]:
        print('======', s)
        token = tokenizer(s, return_attention_mask=False,
                          return_token_type_ids=False)
        tok = {i: torch.tensor([token[i]]) for i in token}
        tok['intent_label_ids'] = None
        tok['slot_labels_ids'] = None

        with torch.no_grad():
            intent_logits, slot_logits = model(**tok)[1:3]
        for i in tok:
            del(i)
            torch.cuda.empty_cache()
        domain = torch.argmax(intent_logits[0]).tolist()
        domain_score = intent_logits[0][domain].cpu().numpy()
        print(domain_label_list[domain], domain_score)
        slots = parse_slots(slot_logits, s, tok['input_ids'])
        print(str(slots))
