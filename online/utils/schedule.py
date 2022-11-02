# /usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import torch

from utils.multi_thread import MultiThreadHelper
from utils.tools import Timer

np.set_printoptions(suppress=True)


class DiscretizedSSP(object):
    def __init__(self, min_step, max_step, grid=2):
        self._min_stepsize = min_step
        self._max_stepsize = max_step
        self._grid = grid
        self._step_pool = self.discretize(self._min_stepsize, self._max_stepsize, self._grid)

    def __getitem__(self, idx):
        return self._step_pool[idx]

    def __len__(self):
        return len(self._step_pool)

    def __str__(self):
        return '[N={}, Grid={}] {}'.format(self.__len__(), self._grid, self._step_pool)

    def discretize(self, min_stepsize, max_stepsize, grid=2):
        step_pool = []
        while (min_stepsize <= max_stepsize):
            step_pool.append(min_stepsize)
            min_stepsize *= grid
        step_pool.append(min_stepsize)
        return np.array(step_pool)


class Schedule(object):
    def __init__(self, ssp, alg, cp_list,
                 use_optimism=False, thread=0,
                 **alg_kwargs):
        self.bases = []

        for i in range(len(ssp)):
            for k in cp_list:
                alg_kwargs[k] = alg_kwargs[k].new()
            self.bases.append(alg(stepsize=ssp[i],
                                  **alg_kwargs))

        self.length = len(ssp)
        self.weight_shape = self.bases[0].model.linear.weight.shape
        self.threads = thread
        self.use_optimism = use_optimism

    def __len__(self):
        return self.length

    def get_x(self):

        output = []
        for i in range(self.length):
            if self.use_optimism:
                output.append(self.bases[i].aux_model.state_dict())
            else:
                output.append(self.bases[i].model.state_dict())

        return output

    def optimism_opt(self, data, target):
        if not self.use_optimism:
            return None

        opt_vector = torch.zeros(self.length)
        loss_vector = torch.zeros(self.length)
        timer = Timer()

        def expert_optimism_opt(idx, expert, data, target):
            torch.set_num_threads(1)

            loss_results = expert.optimism_opt_one_batch(data, target)
            return idx, loss_results['optimism'], loss_results['loss']

        commands = [(expert_optimism_opt, idx, expert, data, target) for idx, expert in enumerate(self.bases)]
        loss_result = MultiThreadHelper(commands, self.threads, multi_process=False)()

        for idx, opt, loss in loss_result:
            opt_vector[idx] = opt['estimate']
            loss_vector[idx] = loss['estimate']

        for expert in self.bases:
            if expert.projection:
                expert.aux_model.project()

        return opt_vector, loss_vector

    def opt(self, data, target, target_data):

        loss_vector = torch.zeros(self.length)

        def expert_opt(idx, expert, data, target):
            estimate_loss, underlying_loss = expert.opt_one_batch(data, target, target_data)
            if expert.projection:
                expert.model.project()

            return idx, estimate_loss

        commands = [(expert_opt, idx, expert, data, target) for idx, expert in enumerate(self.bases)]
        loss_result = MultiThreadHelper(commands, self.threads, multi_process=False)()

        for idx, loss in loss_result:
            if loss is not None:
                loss_vector[idx] = loss

        # project and load to aux
        for expert in self.bases:
            if expert.projection:
                expert.model.project()

        return loss_vector

    def set_func(self, func):
        for item in self.bases:
            item.set_func(func)
