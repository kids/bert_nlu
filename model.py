# -*- coding: utf-8 -*-

import torch
from transformers import BertModel, BertPreTrainedModel
import logging
import torch.nn as nn


class DmBERT(BertPreTrainedModel):
    def __init__(self, config, intent_label_lst, slot_label_lst):
        super(DmBERT, self).__init__(config)
        self.dropout_rate = .1
        self.ignore_index = 0
        self.num_intent_labels = len(intent_label_lst)
        self.num_slot_labels = len(slot_label_lst)
        self.bert = BertModel(config=config)

        self.dropout_intent = nn.Dropout(self.dropout_rate)
        self.linear_intent = nn.Linear(config.hidden_size,
                                       self.num_intent_labels)
        self.dropout_slot = nn.Dropout(self.dropout_rate)
        self.linear_slot = nn.Linear(config.hidden_size,
                                     self.num_slot_labels)

    def forward(self, input_ids, slot_labels_ids, intent_label_ids):
        # sequence_output, pooled_output, (hidden_states), (attentions)
        outputs = self.bert(input_ids,
                            # attention_mask=torch.ones_like(input_ids),
                            # token_type_ids=torch.zeros_like(input_ids)
                            )
        sequence_output = outputs[0]
        pooled_output = outputs[1]  # [CLS]

        intent_logits = self.dropout_intent(pooled_output)
        intent_logits = self.linear_intent(intent_logits)
        slot_logits = self.dropout_slot(sequence_output)
        slot_logits = self.linear_slot(slot_logits)

        intent_loss, slot_loss = 0, 0

        if intent_label_ids is not None:
            if self.num_intent_labels == 1:
                intent_loss_fct = nn.MSELoss()
                intent_loss = intent_loss_fct(intent_logits.view(-1),
                                              intent_label_ids.view(-1))
            else:
                intent_loss_fct = nn.CrossEntropyLoss()
                intent_loss = intent_loss_fct(
                    intent_logits.view(-1, self.num_intent_labels),
                    intent_label_ids.view(-1))

        if slot_labels_ids is not None:
            if torch.sum((slot_labels_ids[:5, :5] > 0).int()) < 5:
                self.ignore_index = 0
            else:
                self.ignore_index = -1
                # self.ignore_index=0
            slot_loss_fct = nn.CrossEntropyLoss(
                ignore_index=self.ignore_index)
            slot_loss = slot_loss_fct(
                slot_logits.view(-1, self.num_slot_labels),
                slot_labels_ids.view(-1))

        outputs = ((intent_loss, slot_loss),
                   intent_logits, slot_logits,
                   pooled_output, sequence_output)

        return outputs
