# /usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import torch.nn.functional as F


class RewritingRisk(object):
    def __init__(self, device, nn_loss=False):
        self.underlying_priors = None
        self.estimate_priors = None
        self.device = device
        self.nn_loss = nn_loss

        if nn_loss:
            print('Non negative loss')

        self.zero_one_loss = False

    def set_priors(self, estimate_priors, underlying_priors=None):
        self.estimate_priors = estimate_priors.to(self.device)
        self.ori_estimate_priors = self.estimate_priors
        if self.nn_loss:
            self.estimate_priors = torch.clamp(self.estimate_priors, min=0)
        if underlying_priors is not None:
            self.underlying_priors = torch.from_numpy(underlying_priors).float().to(self.device)

    def get_priors(self):
        return self.ori_estimate_priors

    def cal_zero_one_loss(self, outputs, target):
        cls_num = len(self.estimate_priors)
        loss_vector = torch.zeros(cls_num, device=self.device)
        predicts = outputs.argmax(-1)
        for i in range(cls_num):
            predicts_i = predicts[target == i]
            loss_vector[i] = (predicts_i != i).to(dtype=torch.float32).mean()

        loss_estimate = self.estimate_priors.dot(loss_vector)
        if self.underlying_priors is None:
            loss_underlying = None
        else:
            loss_underlying = self.underlying_priors.dot(loss_vector)

        return loss_estimate, loss_underlying

    def __call__(self, outputs, target):

        log_soft = F.log_softmax(outputs.float(), 1)
        cls_num = len(self.estimate_priors)
        loss_vector = torch.zeros(cls_num, device=self.device)

        for i in range(cls_num):
            log_soft_i = log_soft[target == i]
            loss_vector[i] = -log_soft_i[:, i].mean()

        loss_estimate = self.estimate_priors.dot(loss_vector)
        if self.underlying_priors is None:
            loss_underlying = None
        else:
            loss_underlying = self.underlying_priors.dot(loss_vector)

        info = {
            'estimate': loss_estimate,
            'underlying': loss_underlying,
        }

        if self.zero_one_loss:
            loss_estimate, loss_underlying = self.cal_zero_one_loss(outputs, target)
            info.update({
                'zo-estimate': loss_estimate,
                'zo-underlying': loss_underlying,
            })

        return info


class ZeroOneRisk(RewritingRisk):
    def __call__(self, outputs, targets):
        cls_num = len(self.estimate_priors)
        confusion_vector = torch.zeros(cls_num, device=self.device)
        for i in range(cls_num):
            confusion_vector[i] = outputs[targets == i][:, i].mean()

        error_vector = 1 - confusion_vector

        loss_estimate = self.estimate_priors.dot(error_vector)

        if self.underlying_priors is None:
            loss_underlying = None
        else:
            loss_underlying = self.underlying_priors.dot(error_vector)

        info = {
            'estimate': loss_estimate,
            'underlying': loss_underlying,
        }
        return info
