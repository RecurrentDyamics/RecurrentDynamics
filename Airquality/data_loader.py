import os
import time

import ujson as json
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

class MySet(Dataset):
    def __init__(self, file_):
        super(MySet, self).__init__()
        self.content = open(file_).readlines()

    def __len__(self):
        return len(self.content)

    def __getitem__(self, idx):
        return self.content[idx]

def collate_fn(recs):
    forward = map(lambda x: json.loads(x)['forward'], recs)
    backward = map(lambda x: json.loads(x)['backward'], recs)

    def to_tensor_dict(recs):
        values = torch.FloatTensor(map(lambda x: x['values'], recs))
        masks = torch.FloatTensor(map(lambda x: x['masks'], recs))
        deltas = torch.FloatTensor(map(lambda x: x['deltas'], recs))

        labels = torch.FloatTensor(map(lambda x: x['labels'], recs))
        label_masks = torch.FloatTensor(map(lambda x: x['label_masks'], recs))

        years = torch.FloatTensor(map(lambda x: x['years'], recs)).long()
        months = torch.FloatTensor(map(lambda x: x['months'], recs)).long()
        days = torch.FloatTensor(map(lambda x: x['days'], recs)).long()
        hours = torch.FloatTensor(map(lambda x: x['hours'], recs)).long()

        return {'values': values, 'masks': masks, 'deltas': deltas, \
                'labels': labels, 'label_masks': label_masks, \
                'years': years, 'months': months, 'days': days, 'hours': hours}

    return {'forward': to_tensor_dict(forward), 'backward': to_tensor_dict(backward)}

def get_loader(filename, batch_size = 64, shuffle = True):
    dataset = MySet(filename)

    data_iter = DataLoader(dataset = dataset, \
                              batch_size = batch_size, \
                              num_workers = 8, \
                              shuffle = shuffle, \
                              pin_memory = True, \
                              collate_fn = collate_fn
    )

    return data_iter
