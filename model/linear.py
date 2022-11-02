# /usr/bin/env python
# -*- coding: utf-8 -*-

from collections import OrderedDict

import torch
from torch import nn


class Base(nn.Module):
    def __init__(self, **kwargs):
        super(Base, self).__init__()
        self.kwargs = kwargs

    def forward(self, x):
        pass

    def get_weights(self):
        record = OrderedDict()
        for name, para in self.named_parameters():
            record[name] = para

        return record

    def get_grad(self):
        record = OrderedDict()
        for name, para in self.named_parameters():
            record[name] = para.grad
        return record

    def new(self):
        return self.__class__(**self.kwargs)


class Linear(Base):
    def __init__(self, input_dim=28 * 28, output_dim=10, R=1):
        super(Linear, self).__init__(input_dim=input_dim, output_dim=output_dim, R=R)
        self.linear = nn.Linear(input_dim, output_dim, bias=False)
        self.R = R
        self.input_dim = input_dim
        self.output_dim = output_dim

        print("Radius: {}".format(self.R))

    def forward(self, x):
        out = x.view(-1, self.num_flat_features(x))
        out = self.linear(out)
        return out

    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

    def project(self):
        if torch.norm(self.linear.weight) > self.R:
            weight = torch.nn.Parameter(self.R * self.linear.weight / torch.norm(self.linear.weight))

            self.linear.weight = weight
