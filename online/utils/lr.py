# /usr/bin/env python
# -*- coding: utf-8 -*-

import math
from abc import ABC, abstractmethod


# # class Lr used for self confidence tuning learning rate # #
class Lr(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def compute_lr(self, **kwargs):
        raise NotImplementedError()


class SelfTuningLr(Lr):
    def __init__(self, scale=1.0, upper_bound=1.):
        self._t = 0
        self._scale = scale
        self._upper_bound = upper_bound

    def compute_lr(self, **kwargs):
        self._t += 1
        return min(self._upper_bound, self._scale / self._t ** 0.5)


class OptSelfTuningLr(Lr):
    def __init__(self, N):
        self.cum_var = 0
        self.N = N

    def compute_lr(self, **kwargs):
        opt_vector = kwargs['opt']
        loss_vector = kwargs['loss']
        diff = (opt_vector - loss_vector).pow(2).max()
        diff = diff.clamp(0, 0.1)
        self.cum_var += diff


        return math.sqrt(math.log(self.N) / (1 + self.cum_var))
