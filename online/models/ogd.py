# /usr/bin/env python
# -*- coding: utf-8 -*-

import math
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
from torch.optim import *
from torch.utils.data import DataLoader

from utils.tools import Timer
from utils.tools import tensor2scalar

__all__ = ["Base", "CustomOGD", "OptOGD"]


class Base(object):
    def __init__(self, cfgs=None, seed=None, **alg_kwargs):
        self.device = alg_kwargs["device"]
        self.model = alg_kwargs['model'].to(self.device)
        self.init = alg_kwargs['init']

        if self.init is not None:
            self.model.load_state_dict(self.init)

        self.cfgs = cfgs
        self.seed = seed
        self.criterion = None
        self.dataloader = DataLoader(alg_kwargs['dataset'],
                                     num_workers=2,
                                     batch_size=alg_kwargs['batch_size'],
                                     shuffle=True,
                                     pin_memory=True)
        self.alg_kwargs = alg_kwargs

        self.cache = []

        for batch_idx, (data, target) in enumerate(self.dataloader):
            data, target = data.to(self.device), target.to(self.device)
            self.cache.append((data, target))

    def set_func(self, func):
        self.criterion = func

    def reinit(self):
        self.__init__(self.cfgs, self.seed, **self.alg_kwargs)

    def opt_one_batch(self, source_data, source_label, target_data):

        output = self.model(source_data)
        info = self.criterion(output.float(), source_label)

        return tensor2scalar(info['estimate']), tensor2scalar(info['underlying'])

    def opt(self, target_data):
        estimate_loss, underlying_loss = 0, 0

        for batch_idx, (source_data, source_label) in enumerate(self.cache):
            _estimate_loss, _underlying_loss = self.opt_one_batch(source_data, source_label, target_data)

            if _estimate_loss is None:
                estimate_loss = None
            else:
                estimate_loss += _estimate_loss

            if _underlying_loss is None:
                underlying_loss = None
            else:
                underlying_loss += _underlying_loss

        if estimate_loss is not None:
            estimate_loss /= len(self.dataloader)
        if underlying_loss is not None:
            underlying_loss /= len(self.dataloader)

        return estimate_loss, underlying_loss

    @torch.no_grad()
    def predict(self, data):
        data = data.to(self.device)
        data = data.to(torch.float32)

        output = self.model(data)
        pred = output.argmax(-1)

        return pred

    def forward(self, target_data, prior_estimate, t):

        pred = self.predict(target_data)
        estimate_loss, underlying_loss = self.opt(target_data)

        return pred, estimate_loss, underlying_loss


class CustomOGD(Base):
    def __init__(self, cfgs=None, seed=None, **algo_kwargs):
        super(CustomOGD, self).__init__(cfgs=cfgs, seed=seed, **algo_kwargs)
        self.lr = algo_kwargs['stepsize']

        self.projection = algo_kwargs["projection"]
        self.optimizer = None
        print('Stepsize: {}'.format(self.lr))

        self.estimate_result = {
            'D': [],
            'G': [],
        }

        self.grad_clip = (self.cfgs is not None) and self.cfgs.get('grad_clip', False)

    def estimate_gd(self):

        _weights = self.model.get_weights()
        weights = torch.cat(tuple(_weights.values()), dim=0)
        D = torch.norm(weights)
        self.estimate_result['D'].append(D.item())
        print('D: {}, Max D: {}'.format(D.item(), np.max(self.estimate_result['D'])))

        try:
            _grads = self.model.get_grad()
            grads = torch.cat(tuple(_grads.values()), dim=0)
            G = torch.norm(grads)
            self.estimate_result['G'].append(G.item())
            print('G: {}, Max G: {}'.format(G.item(), np.max(self.estimate_result["G"])))
        except BaseException:
            pass

    def opt_one_batch(self, source_data, source_label, target_data):
        self.model.train()
        timer = Timer()
        optimizer = SGD(self.model.parameters(), lr=self.lr)
        optimizer.zero_grad()

        output = self.model(source_data)

        info = self.criterion(output.float(), source_label)

        loss = info['estimate']
        loss.backward()

        if self.grad_clip:
            nn.utils.clip_grad_norm_(self.model.parameters(), 1.0, norm_type=1.0)

        optimizer.step()

        return tensor2scalar(info['estimate']), tensor2scalar(info['underlying'])

    def opt(self, target_data):
        estimate_loss, underlying_loss = super().opt(target_data)

        if self.projection:
            self.model.project()

        return estimate_loss, underlying_loss


class OptOGD(CustomOGD):
    def __init__(self, cfgs=None, seed=None, **alg_kwargs):
        super(OptOGD, self).__init__(cfgs, seed, **alg_kwargs)
        self.optimism = alg_kwargs['optimism']
        if self.optimism is not None:
            self.aux_model = self.model.new().to(self.device)
            if self.init is not None:
                self.aux_model.load_state_dict(self.init)

        self.save_grad = alg_kwargs.get('save_grad', False)
        if self.save_grad:
            self.grads = {
                'real': [],
                'optimism': [],
            }

        self.opt_epoch = alg_kwargs['opt_epoch']
        self.opt_lr = alg_kwargs['opt_lr']
        self.opt_optimizer = alg_kwargs["opt_optimizer"]
        print('Opt lr: {}'.format(self.opt_lr))
        print('Opt epoch: {}'.format(self.opt_epoch))
        print("Opt optimizer: {}".format(self.opt_optimizer))

    @torch.no_grad()
    def predict(self, data):
        ori_output = self.model(data)
        ori_pred = ori_output.argmax(-1)
        output = self.aux_model(data)
        pred = output.argmax(-1)

        res = {
            'Ori': ori_pred,
            'Opt': pred
        }

        if (ori_pred != pred).any():
            print('Predict diff: {} / {}'.format((ori_pred != pred).sum(), len(data)))

        return res

    def opt_one_batch(self, source_data, source_label, target_data):
        # update model: \hat{w}_{t+1} = \hat{w}_t - \eta \nabla R(w_t)
        output = self.aux_model(source_data)
        info = self.criterion(output.float(), source_label)

        loss = info['estimate']
        loss.backward()

        grads = self.aux_model.get_grad()
        weights = self.model.get_weights()

        for name, grad in grads.items():
            new_weight = weights[name].add(grad, alpha=-self.lr)
            weights[name] = nn.Parameter(new_weight)
        self.model.load_state_dict(weights)

        return None, None

    def cal_dist(self, weights_a, weights_b):
        _weights_a = torch.cat(tuple(weights_a.values()), dim=0)
        _weights_b = torch.cat(tuple(weights_b.values()), dim=0)

        return torch.dist(_weights_a, _weights_b, p=2).pow(2)

    def opt_LBFGS(self, source_data, source_label, criterion, base_weights):
        optimizer = LBFGS(self.aux_model.parameters(), lr=self.opt_lr, max_iter=self.opt_epoch)

        def eval():
            optimizer.zero_grad()
            output = self.aux_model(source_data)
            info = criterion(output.float(), source_label)

            loss_data = info['estimate']

            curr_weights = self.aux_model.get_weights()
            loss_reg = self.cal_dist(curr_weights, base_weights)

            loss = self.lr * loss_data + 0.5 * loss_reg
            loss.backward()

            return loss

        optimizer.step(eval)

    def opt_SGD(self, source_data, source_label, criterion, base_weights):
        optimizer = SGD(self.aux_model.parameters(),
                        lr=self.opt_lr,
                        momentum=0.01,
                        nesterov=True)

        history_loss, count = 0, 0

        timer = Timer()
        timer.tik()

        for i in range(self.opt_epoch):
            optimizer.zero_grad()
            output = self.aux_model(source_data)
            info = criterion(output.float(), source_label)

            loss_data = info['estimate']

            curr_weights = self.aux_model.get_weights()
            loss_reg = self.cal_dist(curr_weights, base_weights)

            loss = self.lr * loss_data + 0.5 * loss_reg
            loss.backward()

            optimizer.step()

            if math.isclose(history_loss, loss, abs_tol=1e-6):
                count += 1
            if count > 5:
                break
            history_loss = loss


    def optimism_opt_one_batch(self, source_data, source_label):
        # update aux_model: w_t = \argmin \eta H_t{w} + \frac{1}{2} \|w - \hat{w}_t\|_2

        criterion = self.optimism.get_criterion()
        base_weights = self.model.state_dict()
        self.aux_model.load_state_dict(base_weights)

        if self.opt_optimizer == 'LBFGS':
            self.opt_LBFGS(source_data, source_label, criterion, base_weights)
        elif self.opt_optimizer == 'SGD':
            self.opt_SGD(source_data, source_label, criterion, base_weights)
        else:
            raise NotImplementedError
        # H_t(w)
        with torch.no_grad():
            output = self.aux_model(source_data)
            opt_info = criterion(output.float(), source_label)
            loss_info = self.criterion(output.float(), source_label)

        results = {
            'optimism': opt_info,
            'loss': loss_info
        }

        return results

    def loss_mean(self, loss_info):
        result = defaultdict(lambda: defaultdict(int))
        for res in loss_info:
            for k1, v1 in res.items():
                for k2, v2 in v1.items():
                    if v2 is None:
                        result[k1][k2] = None
                    else:
                        result[k1][k2] += v2

        for k1, v1 in result.items():
            for k2, v2 in v1.items():
                if result[k1][k2] is not None:
                    result[k1][k2] /= len(loss_info)

        return result

    def optimism_opt(self, target_data, prior_estimate, t):
        self.set_optimism_info(target_data, prior_estimate, t)
        estimate_loss, underlying_loss = 0, 0
        opt = 0

        loss_info = []
        for batch_idx, (source_data, source_label) in enumerate(self.cache):
            res = self.optimism_opt_one_batch(source_data, source_label)
            _estimate_loss = res['loss']['estimate']
            _underlying_loss = res['loss']['underlying']
            _opt = res['optimism']['estimate']
            loss_info.append(res)

            opt += _opt
            estimate_loss += _estimate_loss
            if _underlying_loss is None:
                underlying_loss = None
            else:
                underlying_loss += _underlying_loss

        opt /= len(self.dataloader)
        estimate_loss /= len(self.dataloader)
        if underlying_loss is not None:
            underlying_loss /= len(self.dataloader)

        loss_info = self.loss_mean(loss_info)

        if self.projection:
            self.aux_model.project()

        return estimate_loss, underlying_loss, opt, loss_info

    def set_optimism_info(self, target_data, prior_estimate, t):
        self.optimism.set_info(target_data, prior_estimate, t)


    def forward(self, target_data, prior_estimate, t):
        _estimate_loss, _underlying_loss, opt, _loss_info = self.optimism_opt(target_data, prior_estimate, t)
        pred, estimate_loss, underlying_loss = \
            super(OptOGD, self).forward(target_data, prior_estimate, t)

        self.opt_value = opt

        return pred, _estimate_loss, _underlying_loss

    def get_opt_value(self):
        return self.opt_value
