# coding=utf-8
import json
import torch
import torch.nn as nn
from flask import Flask, request
from transformers import BertConfig, BertTokenizer, BertPreTrainedModel, BertModel

app = Flask(__name__)

mdl_path = './'

cfg = BertConfig.from_pretrained(mdl_path)
max_length = cfg.max_position_embeddings
padding_id = cfg.pad_token_id


domain_map = {}
domain_label_list = []
for i, s in enumerate([
    'domain.op.control',
    'domain.op.msgcall',
    'domain.op.app',
    'domain.op.geonavi',
    'domain.op.media.music',
    'domain.op.media.fm',
    'domain.op.media.video',
    'domain.op.media.news',
    'domain.op.booking',
    'domain.op.other',
    # 'domain.dialog.salut',
    # 'domain.dialog.chat',
    'domain.dialog.complain',
    # 'domain.dialog.manual',
    'domain.dialog.weather',
    'domain.dialog.lbs',
    # 'domain.dialog.traffic',
    # 'domain.dialog.status',
    'domain.dialog.kgsearch',
    'domain.dialog.other',
    # 'domain.fillslot',
        'domain.non', ]):
    domain_map[s] = i
    domain_label_list.append(s)
domain_map['domain.dialog.salut'] = domain_map['domain.dialog.other']
domain_map['domain.dialog.chat'] = domain_map['domain.dialog.other']
domain_map['domain.dialog.status'] = domain_map['domain.dialog.other']
domain_map['domain.dialog.manual'] = domain_map['domain.dialog.other']
domain_map['domain.dialog.traffic'] = domain_map['domain.dialog.lbs']

slot_map = {
    'O': 0,
}
slot_label_list = ['O']
for i, s in enumerate([
    'slot.number',
    'slot.datetime',
    'slot.object.app',
    'slot.object.device',
    'slot.object.geoloc',
    'slot.object.person',
    'slot.object.ipname',
    'slot.object.other',
    'slot.property',
    # 'slot.expression',
        'slot.move', ]):
    slot_map[s] = i + 1
    slot_label_list.extend([s, s])
slot_map['slot.object.ipname.book'] = slot_map['slot.object.ipname']
slot_map['slot.object.ipname.song'] = slot_map['slot.object.ipname']
slot_map['slot.object.ipname.motionpic'] = slot_map['slot.object.ipname']
slot_map['slot.object.ipname.podcast'] = slot_map['slot.object.ipname']
slot_map['slot.object.app.appname'] = slot_map['slot.object.app']
slot_map['slot.object.app.apppage'] = slot_map['slot.object.app']


class DmBERT(BertPreTrainedModel):
    def __init__(self, config, intent_label_lst, slot_label_lst):
        super(DmBERT, self).__init__(config)
        self.dropout_rate = .1
        self.ignore_index = 0
        self.num_intent_labels = len(intent_label_lst)
        self.num_slot_labels = len(slot_label_lst)
        self.bert = BertModel(config=config)

        self.dropout_intent = nn.Dropout(self.dropout_rate)
        self.linear_intent = nn.Linear(config.hidden_size, self.num_intent_labels)
        self.dropout_slot = nn.Dropout(self.dropout_rate)
        self.linear_slot = nn.Linear(config.hidden_size, self.num_slot_labels)

    def forward(self, input_ids, slot_labels_ids, intent_label_ids):
        outputs = self.bert(input_ids,
                            # attention_mask=torch.ones_like(input_ids),
                            # token_type_ids=torch.zeros_like(input_ids)
                            )  # sequence_output, pooled_output, (hidden_states), (attentions)
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
                intent_loss = intent_loss_fct(intent_logits.view(-1), intent_label_ids.view(-1))
            else:
                intent_loss_fct = nn.CrossEntropyLoss()
                intent_loss = intent_loss_fct(intent_logits.view(-1, self.num_intent_labels),
                                              intent_label_ids.view(-1))

        if slot_labels_ids is not None:
            if torch.sum((slot_labels_ids[:5, :5] > 0).int()) < 5:
                self.ignore_index = 0
            else:
                self.ignore_index = -1
                # self.ignore_index=0
            slot_loss_fct = nn.CrossEntropyLoss(ignore_index=self.ignore_index)
            slot_loss = slot_loss_fct(slot_logits.view(-1, self.num_slot_labels),
                                      slot_labels_ids.view(-1))

        outputs = ((intent_loss, slot_loss), intent_logits, slot_logits, pooled_output, sequence_output)

        return outputs  # (loss), logits, (hidden_states), (attentions) # Logits tuple of intent and slot


def parse_slots(slot_logits, s, tok, slot_logit_thresh=1.9):  # 2.9 for 3cat data
    token_len = len(slot_logits[0])
    max_v, max_i = torch.max(slot_logits[0], 1)
    # print(token_len,max_v[:token_len],max_i[:token_len])
    # max_i[max_v<slot_logit_thresh]=0
    slot_bid = (max_i[:token_len] % 2 == 1).nonzero().reshape(-1).tolist()
    slot_bid.append(token_len)
#     print(slot_bid)
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
            if i == len(l) - 1:
                rs.append(s)
                break
            n = s.find(tokenizer.ids_to_tokens[int(l[i + 1])])
        else:
            n = 1
        rs.append(s[:n])
        s = s[n:]
    rs = ['|'] + rs + ['|']

    nret = []
    for cret in ret:
        cur_slot_score = torch.mean(max_v[cret[1]:cret[2]]).detach().numpy().tolist()
        cur_slot_text = ''.join(rs[cret[1]:cret[2]])
        nret.append([cret[0], cur_slot_text, cur_slot_score])

    return nret


@app.route('/')
def index():
    return 'Directly go request RestFul api'


@app.route('/dm', methods=['GET', ])
def skills():
    s = request.args['str']
    # enc=s.split('\n')
    token = tokenizer(s, return_attention_mask=False, return_token_type_ids=False)
    tok = {i: torch.tensor([token[i]]) for i in token}
    tok['intent_label_ids'] = None
    tok['slot_labels_ids'] = None

    intent_logits, slot_logits = model(**tok)[1:3]
    domain_name = domain_label_list[torch.argmax(intent_logits[0]).tolist()]
    domain_score = torch.max(intent_logits[0]).detach().numpy().tolist()

    slots = parse_slots(slot_logits, s, tok['input_ids'])
    return domain_name + ':' + str(domain_score) + ' = ' + str(slots)  # .encode('utf-8')


if __name__ == '__main__':
    print("Loading PyTorch model and Flask starting server ...")
    tokenizer = BertTokenizer.from_pretrained(mdl_path, config=cfg)
    # model = DmBERT.from_pretrained(mdl_path,
    #                               config=cfg,
    #                               intent_label_lst=domain_label_list,
    #                               slot_label_lst=slot_label_list)
    model = DmBERT(cfg, intent_label_lst=domain_label_list, slot_label_lst=slot_label_list)
    model = torch.quantization.quantize_dynamic(model, dtype=torch.qint8)
    model = torch.load(mdl_path + 'pytorch_model.bin')
    # model.eval()
    app.run(host='0.0.0.0', port='8080')
