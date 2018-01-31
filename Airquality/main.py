import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np

import time
import utils
import models
import argparse
import data_loader
import pandas as pd
import ujson as json

from sklearn import metrics

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type = int, default = 100)
parser.add_argument('--batch_size', type = int, default = 64)
parser.add_argument('--model', type = str)
parser.add_argument('--train_file', type = str)
parser.add_argument('--eval_file', type = str)
args = parser.parse_args()

config = json.load(open('./config'))

train_iter = data_loader.get_loader(args.train_file, batch_size = args.batch_size)
val_iter = data_loader.get_loader(args.eval_file, batch_size = args.batch_size)

def train(model):
    optimizer = optim.Adam(model.parameters(), lr = 1e-3)

    for epoch in xrange(args.epochs):
        model.train()

        running_loss = 0.0

        for idx, data in enumerate(train_iter):
            data = utils.to_var(data)
            ret = model.run_on_batch(data, optimizer)
            running_loss += ret['g_loss'].data[0]
            print '\r Progress epoch {}, {:.2f}%, average loss {}'.format(epoch, (idx + 1) * 100.0 / len(train_iter), running_loss / (idx + 1.0)),

        if epoch % 10 == 0:
            evaluate(model, val_iter, epoch = epoch)

def evaluate(model, val_iter, epoch = 0):
    def eval_parse(ret):
        masks = ret['label_masks'].cpu().numpy()

        years = ret['years'].unsqueeze(dim = 2).expand(masks.shape[0], 36, 36).cpu().numpy()
        months = ret['months'].unsqueeze(dim = 2).expand(masks.shape[0], 36, 36).cpu().numpy()
        days = ret['days'].unsqueeze(dim = 2).expand(masks.shape[0], 36, 36).cpu().numpy()
        hours = ret['hours'].unsqueeze(dim = 2).expand(masks.shape[0], 36, 36).cpu().numpy()

        predictions = ret['predictions'].cpu().numpy() * config['std'] + config['mean']
        predictions = predictions[np.where(masks == 1)]


        years = years[np.where(masks == 1)] + 2014
        months = months[np.where(masks == 1)] + 1
        days = days[np.where(masks == 1)] + 1
        hours = hours[np.where(masks == 1)]

        labels = ret['labels'].cpu().numpy() * config['std'] + config['mean']
        labels = labels[np.where(masks == 1)]

        stations = torch.FloatTensor(range(1, 36 + 1)).view(1, 1, 36)
        stations = stations.expand(masks.shape[0], 36, 36).numpy()

        stations = stations[np.where(masks == 1)]

        df['years'] += years.tolist()
        df['months'] += months.tolist()
        df['days'] += days.tolist()
        df['hours'] += hours.tolist()
        df['stations'] += stations.tolist()
        df['predictions'] += predictions.tolist()
        df['labels'] += labels.tolist()


    model.eval()
    df = {'years': [], 'months': [], 'days': [], 'hours': [], 'stations': [], 'predictions': [], 'labels': []}
    for idx, data in enumerate(val_iter):
        data = utils.to_var(data)
        ret = model.run_on_batch(data, optimizer = None)
        eval_parse(ret)
    df = pd.DataFrame(df).groupby(['years', 'months', 'days', 'hours', 'stations']).agg('mean').reset_index()
    df.to_csv('./imputation.csv')

    print 'Eval loss {}'.format((df['predictions'] - df['labels']).abs().mean())

def run():
    model = getattr(models, args.model).Model()

    if torch.cuda.is_available():
        model = model.cuda()

    train(model)

if __name__ == '__main__':
    run()
