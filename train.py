# -*- coding: utf-8 -*-

import torch
import os
import logging

from transformers import AdamW, get_linear_schedule_with_warmup

from torch.utils.data import TensorDataset, DataLoader, RandomSampler


def start_from_pretrained(pre_mdl_path, model):
    cp_mdl = BertModel.from_pretrained(pre_mdl_path)
    a, b = model.state_dict(), cp_mdl.state_dict()
    for i in a:
        if i[5:] not in b:
            continue
        print(i[5:], a[i].shape, b[i[5:]].shape)
        if len(a[i].shape) == 2:
            m, n = a[i].shape
            a[i] = b[i[5:]][:m, :n]
        else:
            a[i] = b[i[5:]]


def save_model(model_dir):
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    model_to_save = model.module if hasattr(model, 'module') else model
    model_to_save.save_pretrained(model_dir)


def train_model(model, dataset, save_path, device='cuda'):

    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters()
         if not any(nd in n for nd in no_decay)],
         'weight_decay': 0.0},
        {'params': [p for n, p in model.named_parameters()
         if any(nd in n for nd in no_decay)],
         'weight_decay': 0.0}
    ]

    # optimizer = AdamW(optimizer_grouped_parameters, lr=2e-3, eps=1e-8)
    optimizer = torch.optim.SGD(model.parameters(), lr=2e-1)
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=10,
                                                num_training_steps=1000)

    model.dropout_rate = 0
    model.to(device)
    model.train()

    sampler = RandomSampler(dataset)
    batch_size = 178

    gradient_accumulation_steps = 1
    global_step = 0
    model.zero_grad()
    loss_plot = []
    tr_loss = 0.0
    tr_loss_i, tr_loss_s = 0.0, 0.0

    for _ in range(1):
        train_dataloader = DataLoader(dataset,
                                      sampler=sampler,
                                      batch_size=batch_size)
        for step, batch in enumerate(train_dataloader):
            print(',', end='')

            batch = tuple(t.to(device) for t in batch)
            y = batch[2].squeeze().detach().cpu().numpy().tolist()

            inputs = {'input_ids': batch[0],
                      'slot_labels_ids': batch[1],
                      'intent_label_ids': batch[2]}
            outputs = model(**inputs)
            del(batch)

            intent_loss, slot_loss = outputs[0]
            (intent_loss * 0.1 + slot_loss * 0.2).backward()

            tr_loss_i += intent_loss.item()
            tr_loss_s += slot_loss.item()
            if (step + 1) % gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1

                if global_step % 20 == 0:
                    intent_logit = outputs[1]
                    prate = (sum([1 if iy == iz else 0 for iy, iz in
                                  zip(y,
                                      torch.argmax(
                                          intent_logit, 1).tolist())])
                             ) / len(y)
                    print(tr_loss_i / global_step,
                          tr_loss_s / global_step, prate)
                    if global_step % 500 == 0:
                        print(time.ctime(), 'saving..')
                        save_model(save_path)
