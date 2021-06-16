# -*- coding: utf-8 -*-

from torch.utils.data import TensorDataset
import torch


def mk_tgts():
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
    return domain_map, domain_label_list, slot_map, slot_label_list


def xml2tsr(a, nslot, tokenizer):
    # begin temp filtering for additive training
    #     if 'location_function' not in a:
    #         if random.random()>0.2:
    #             return [],[]
    # end temp filtering for additive training
    sx, sl = [101, ], [0, ]
    for chunk in a.split('<'):
        if '>' in chunk:
            cx = tokenizer(chunk.split('>')[1])['input_ids'][1:-1]
            if '/' in chunk:
                cl = [0] * (len(cx))
                # print('0',chunk,cx,cl)
            else:
                slt = chunk.split('>')[0]
                if slt not in nslot:
                    cl = [0] * (len(cx))
                else:
                    cl = [nslot[slt] * 2 - 1] + [nslot[slt] * 2] * (len(cx) - 1)
        else:
            cx = tokenizer(chunk)['input_ids'][1:-1]
            cl = [0] * (len(cx))
            # print('1',chunk,cx,cl)
        sx.extend(cx)
        sl.extend(cl)
    return sx + [102], sl + [0]


def mk_dataset(raw_datafile, dataset_file, tokenizer):

    if os.path.exist(dataset_file):
        return torch.load(dataset_file)

    keep_max_length = 15
    all_input_sequence, all_slot_labels_ids, all_domain_label = [], [], []

    lbl_cnt = []
    i = 0
    pre = []
    with open(raw_datafile) as f:
        for l in f:
            i += 1
            if i % 10000 == 0:
                print(i, end=',')
            if 'fillslot' in l:
                continue
            domain, s = l.strip().split('|')
            # s=re.findall(r'</(.*?)>',l)
            i_input, i_label = xml2tsr(s, slot_map, tokenizer)
            if i_input == pre:
                # print(l)
                pass
            else:
                pre = i_input

            if len(i_input) > keep_max_length:
                i_input = i_input[:keep_max_length]
                i_label = i_label[:keep_max_length]

            lbl_cnt.extend(i_label)
            if len(i_input) != len(i_label):
                print(l, i_input, i_label)
                continue
            all_input_sequence.append(
                i_input + [padding_id] * (keep_max_length - len(i_input)))
            all_slot_labels_ids.append(
                i_label + [padding_id] * (keep_max_length - len(i_input)))
            all_domain_label.append([domain_map[domain]])

    dataset = TensorDataset(
        torch.tensor(all_input_sequence, dtype=torch.long),
        torch.tensor(all_slot_labels_ids, dtype=torch.long),
        torch.tensor(all_domain_label, dtype=torch.long)
    )
    torch.save(dataset, dataset_file)
