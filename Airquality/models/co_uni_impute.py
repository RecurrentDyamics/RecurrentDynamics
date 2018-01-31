import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.autograd import Variable
from torch.nn.parameter import Parameter

import math
import utils
import argparse
import data_loader

from sklearn import metrics

SEQ_LEN = 36
RNN_HID_SIZE = 64

class TemporalDecay(nn.Module):
    def __init__(self, input_size):
        super(TemporalDecay, self).__init__()
        self.build(input_size)

    def build(self, input_size):
        self.W = Parameter(torch.Tensor(input_size, input_size))
        self.b = Parameter(torch.Tensor(input_size))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.W.size(0))
        self.W.data.uniform_(-stdv, stdv)
        if self.b is not None:
            self.b.data.uniform_(-stdv, stdv)

    def forward(self, d):
        gamma = F.relu(F.linear(d, self.W / torch.norm(self.W), self.b / torch.norm(self.b)))
        gamma = torch.exp(-gamma)
        return gamma


class FeatureRegression(nn.Module):
    def __init__(self, input_size, extra_size):
        super(FeatureRegression, self).__init__()
        self.build(input_size, extra_size)

    def build(self, input_size, extra_size):
        self.W = Parameter(torch.Tensor(input_size, input_size))
        self.b = Parameter(torch.Tensor(input_size))

        m = torch.ones(input_size, input_size) - torch.eye(input_size, input_size)
        self.register_buffer('m', m)

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.W.size(0))
        self.W.data.uniform_(-stdv, stdv)
        if self.b is not None:
            self.b.data.uniform_(-stdv, stdv)

    def forward(self, x):
        z_h = F.linear(x, self.W * Variable(self.m), self.b)
        return z_h


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.build()

    def build(self):
        self.em_year = nn.Embedding(2, 2)
        self.em_month = nn.Embedding(12, 2)
        self.em_day = nn.Embedding(32, 3)
        self.em_hour = nn.Embedding(24, 3)

        self.rnn_cell = nn.LSTMCell(36 * 3, RNN_HID_SIZE)

        self.history_regression = nn.Linear(RNN_HID_SIZE + 10, 36)
        self.feature_regression = FeatureRegression(36, 10)

        self.temp_decay = TemporalDecay(input_size = 36)
        self.weight_combine = nn.Linear(36 * 2, 36)

    def get_date(self, year, month, day, hour):
        year = self.em_year(year.contiguous().view(-1, 1)).squeeze()
        month = self.em_month(month.contiguous().view(-1, 1)).squeeze()
        day = self.em_day(day.contiguous().view(-1, 1)).squeeze()
        hour = self.em_hour(hour.contiguous().view(-1, 1)).squeeze()

        return torch.cat([year, month, day, hour], dim = 1)

    def forward(self, data, alpha = 0.0):
        # Original sequence with 24 time steps
        values = data['values']
        masks = data['masks']
        deltas = data['deltas']

        labels = data['labels']
        label_masks = data['label_masks']

        years = data['years'] - 2014
        months = data['months'] - 1
        days = data['days'] - 1
        hours = data['hours']

        h = Variable(torch.zeros((values.size()[0], RNN_HID_SIZE)))
        c = Variable(torch.zeros((values.size()[0], RNN_HID_SIZE)))

        if torch.cuda.is_available():
            h, c = h.cuda(), c.cuda()

        g_loss = 0.0

        predictions = []

        for t in range(SEQ_LEN):
            year, month, day, hour = years[:, t], months[:, t], days[:, t], hours[:, t]

            date = self.get_date(year, month, day, hour)

            x_h = self.history_regression(torch.cat([date, h], dim = 1))

            x = values[:, t, :]
            m = masks[:, t, :]
            d = deltas[:, t, :]

            g_loss += torch.sum(torch.abs(x - x_h) * m) / torch.sum(m)

            x_c =  m * x +  (1 - m) * x_h

            gamma = self.temp_decay(d)
            beta = F.sigmoid(self.weight_combine(torch.cat([gamma, m], dim = 1)))

            z_h = self.feature_regression(x_c)
            g_loss += torch.sum(torch.abs(x - z_h) * m) / torch.sum(m)

            c_h = beta * z_h + (1 - beta) * x_h
            g_loss += torch.sum(torch.abs(x - c_h) * m) / torch.sum(m)

            predictions.append(c_h.unsqueeze(dim = 1))

            c_c = m * x + (1 - m) * c_h

            inputs = torch.cat([c_c, gamma, m], dim = 1)

            h, c = self.rnn_cell(inputs, (h, c))

        predictions = torch.cat(predictions, dim = 1)

        return {'g_loss': g_loss / SEQ_LEN, 'predictions': predictions.data, 'labels': labels.data, \
                'years': years.data, 'months': months.data, 'days': days.data, 'hours': hours.data, \
                'masks': masks.data, 'label_masks': label_masks.data}


    def run_on_batch(self, data, optimizer):
        ret = self(data['forward'])

        if optimizer is not None:
            optimizer.zero_grad()
            ret['g_loss'].backward()
            optimizer.step()

        return ret
