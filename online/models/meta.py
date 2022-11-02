# /usr/bin/env python
# -*- coding: utf-8 -*-

import torch

from online.utils.lr import SelfTuningLr, OptSelfTuningLr


class Hedge(object):
    def __init__(self, N, lr, prior, use_optimism=False):
        self._N = N
        self._lr = lr
        self.prob = self.set_prior(prior)
        self.use_optimism = use_optimism
        self.t = 0
        if use_optimism:
            self.aux_prob = self.set_prior(prior)

    def set_prior(self, prior):
        if prior == 'uniform':
            prob = [1 / self._N for i in range(self._N)]
        elif prior == 'nonuniform':
            prob = [(self._N + 1) / (self._N * i * (i + 1)) for i in range(1, self._N + 1)]
        else:
            prob = prior

        return torch.tensor(prob)

    @torch.no_grad()
    def opt(self, loss):

        exp_loss = torch.exp(-self.lr * loss)
        ori_prob = self.prob.detach().clone()
        self.prob *= exp_loss
        self.prob /= self.prob.sum()
        if torch.any(torch.isnan(self.prob)):
            self.prob = ori_prob

    @torch.no_grad()
    def optimism_opt(self, opt):

        exp_loss = torch.exp(-self.lr * opt)
        self.aux_prob = self.prob * exp_loss
        self.aux_prob /= self.aux_prob.sum()

        if torch.any(torch.isnan(self.aux_prob)):
            self.aux_prob = self.prob

    def get_prob(self):
        if self.use_optimism:
            return self.aux_prob
        else:
            return self.prob

    def update_lr(self, **kwargs):
        lr = self._lr
        if isinstance(lr, SelfTuningLr) or \
                isinstance(lr, OptSelfTuningLr):
            lr = lr.compute_lr(**kwargs)

        self.lr = lr
