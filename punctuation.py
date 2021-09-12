#!/usr/bin/env python

import json
import torch
from transformers import BertTokenizer
from model import BertFunc
from pathlib import Path

bert_fpath = Path('models/hfl/chinese-macbert-base')
model_fpath = Path('models/20210825_164251')
hparams_fpath = model_fpath / 'hyperparameters.json'

data = Path('data.scp.txt')

hparams = json.loads(hparams_fpath.read_text())

tokenizer = BertTokenizer.from_pretrained(bert_fpath, do_lower_case = True,)

punctuation_enc = {'O': 0, '，': 1, '。': 2, '？': 3}
punctuations = list(punctuation_enc.keys())
segment_size = 200

dropout = 0.
model = BertFunc(segment_size, len(punctuations), 0., None)
model_params = torch.load(model_fpath / 'model')
model.load_state_dict(model_params)
model.cuda()
model.eval()

lines = data.read_text().split('\n')
texts = [line.split(' ', 1)[1] for line in lines if len(line.split(' ', 1)) > 1]

tokenids = tokenizer(
    texts,
    add_special_tokens = False,
    max_length = 64,
    padding = 'max_length',
    truncation = True,
    return_tensors = 'pt',
)['input_ids'].cuda()


punctuations[0] = ''
with torch.no_grad():
    for idx, line in enumerate(tokenids):
        pred = model(line.unsqueeze(0)).argmax(dim = 1)
        line = texts[idx]
        pred = pred[:len(line)]
        sent = []
        for c, p in zip(line, pred.cpu().tolist()):
            sent.append(c)
            sent.append(punctuations[p])

        txt = ''.join(sent)
        print(txt)
