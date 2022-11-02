# /usr/bin/env python
# -*- coding: utf-8 -*-

from abc import ABC
from functools import reduce

import numpy as np
import torch


class BaseHint(ABC):
    def __init__(self, loss_func, **kwargs):
        self.loss_func = loss_func
        self.kwargs = kwargs
        self.device = kwargs.get('device', 'cpu')
        print(kwargs)

    def get_criterion(self):
        return self.loss_func

    def get_priors(self):
        return self.loss_func.get_priors()

    def set_info(self, target_data, prior_estimate, t):
        pass

    def get_grad(self, data, target, model):
        output = model(data)
        info = self.loss_func(output.float(), target)
        loss = info['estimate']
        loss.backward()

        return model.get_grad()


class FwdHint(BaseHint):
    def __init__(self, loss_func, **kwargs):
        super(FwdHint, self).__init__(loss_func, **kwargs)

    def set_info(self, target_data, prior_estimate, t):
        self.loss_func.set_priors(prior_estimate)


class WinHint(BaseHint):
    def __init__(self, loss_func, **kwargs):
        super(WinHint, self).__init__(loss_func, **kwargs)
        self.win_len = kwargs.get("win_len", 10)
        self.window = []  # queue for history information

    def set_info(self, target_data, prior_estimate, t):
        self.window.append(prior_estimate)
        if len(self.window) > self.win_len:
            self.window.pop(0)

        prior_mean = reduce(lambda x, y: x + y, self.window) / len(self.window)

        self.loss_func.set_priors(prior_mean)


class PeriHint(BaseHint):
    def __init__(self, loss_func, **kwargs):
        super(PeriHint, self).__init__(loss_func, **kwargs)
        self.q_len = kwargs.get("q_len", 200)
        self.cls_num = kwargs.get("cls_num", 3)
        self.queue = torch.zeros((self.q_len, self.cls_num),
                                 dtype=torch.float32, device=self.device)
        self.mark = torch.zeros(self.q_len, dtype=torch.bool)
        self.num = torch.zeros(self.q_len, dtype=torch.int64)
        self.alpha = 0.5

    def set_info(self, target_data, prior_estimate, t):
        index = t % self.q_len
        prior_estimate = prior_estimate.to(self.device)
        self.num[index] += 1
        self.alpha = 1 / self.num[index]

        if self.mark[index]:
            self.queue[index] = self.queue[index] * (1 - self.alpha) + prior_estimate * self.alpha
        else:
            self.queue[index] = prior_estimate
            self.mark[index] = True

        self.loss_func.set_priors(self.queue[index])


class OKMHint(BaseHint):
    def __init__(self, loss_func, **kwargs):
        super(OKMHint, self).__init__(loss_func, **kwargs)
        self.k = kwargs.get("k", 10)
        self.cls_num = kwargs.get("cls_num", 3)
        self.dim = kwargs.get('dim', 12)
        self.alpha = kwargs.get('alpha', 0.1)
        self.decay = kwargs.get('decay', True)  # whether use discount

        self.rng = kwargs.get("rng", None)
        if self.rng == None:
            self.rng = np.random.default_rng()

        self.centers = torch.zeros((self.k, self.cls_num), dtype=torch.float32, device=self.device)
        self.mark = torch.zeros(self.k, dtype=torch.bool)
        self.prototype = torch.zeros((self.k, self.dim), dtype=torch.float32, device=self.device)
        if not self.decay:
            self.num = torch.zeros(self.k, dtype=torch.int64)

    def set_info(self, target_data, prior_estimate, t):
        target_data, prior_estimate = target_data.to(self.device), prior_estimate.to(self.device)
        target_data_mean = target_data.mean(dim=0).view(1, -1)

        if (self.mark == False).sum() > 0:
            candidate = (self.mark == False).nonzero(as_tuple=True)[0]
        else:
            dists = torch.cdist(self.prototype, target_data_mean, p=2).view(-1)
            candidate = (dists == dists.min()).nonzero(as_tuple=True)[0]

        idx = candidate[self.rng.integers(len(candidate))]

        if not self.decay:
            self.num[idx] += 1
            self.alpha = 1 / self.num[idx]

        if self.mark[idx]:
            self.centers[idx] = self.centers[idx] * (1 - self.alpha) + \
                                prior_estimate * self.alpha
            self.prototype[idx] = self.prototype[idx] * (1 - self.alpha) + \
                                  target_data_mean.view(-1) * self.alpha
        else:
            self.centers[idx] = prior_estimate
            self.prototype[idx] = target_data_mean.view(-1)
            self.mark[idx] = True

        self.loss_func.set_priors(self.centers[idx])
