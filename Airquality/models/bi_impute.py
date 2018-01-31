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

import uni_impute
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
        gamma = F.relu(F.linear(d, self.W, self.b))
        gamma = torch.exp(-gamma)
        return gamma

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.build()

    def build(self):
        self.impute_f = uni_impute.Model()
        self.impute_b = uni_impute.Model()

    def forward(self, data):
        ret_f = self.impute_f(data['forward'])
        ret_b = self.reverse(self.impute_b(data['backward']))

        ret = self.merge_ret(ret_f, ret_b)

        return ret

    def merge_ret(self, ret_f, ret_b):
        g_loss_f = ret_f['g_loss']
        g_loss_b = ret_b['g_loss']
        g_loss_c = self.get_consistency_g_loss(ret_f['predictions'], ret_b['predictions'])

#        g_loss = g_loss_f + g_loss_b + g_loss_c
        g_loss = g_loss_f + g_loss_b

        predictions = (ret_f['predictions'] + ret_b['predictions']) / 2

        ret_f['g_loss'] = g_loss
        ret_f['predictions'] = predictions

        return ret_f

    def get_consistency_g_loss(self, pred_f, pred_b):
        g_loss = torch.pow(pred_f - pred_b, 2).mean()
        return g_loss

    def reverse(self, ret):
        def reverse_tensor(tensor_):
            if tensor_.dim() <= 1:
                return tensor_
            indices = range(tensor_.size()[1])[::-1]
            indices = torch.LongTensor(indices)

            if torch.cuda.is_available():
                indices = indices.cuda()

            return tensor_.index_select(1, indices)

        for key in ret:
            ret[key] = reverse_tensor(ret[key])

        return ret

    def run_on_batch(self, data, optimizer):
        ret = self(data)

        if optimizer is not None:
            optimizer.zero_grad()
            ret['g_loss'].backward()
            optimizer.step()

        return ret

