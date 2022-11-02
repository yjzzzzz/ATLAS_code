# /usr/bin/env python
# -*- coding: utf-8 -*-

from abc import ABC, abstractmethod

import numpy as np


def parse_prior(priors):
    if isinstance(priors, int):
        priors = np.ones(priors) / priors
    elif isinstance(priors, str):
        values = priors.split('@')
        target, total = int(values[0]), int(values[1])
        priors = np.zeros(total)
        priors[target] = 1

    return priors


class BaseShift(ABC):
    def __init__(self, q1, q2=None, T=None, rng=None):
        if rng is None:
            self.rng = np.random.RandomState()
        else:
            self.rng = rng

        self.q1 = np.array(parse_prior(q1))
        self.q2 = np.array(parse_prior(q2))
        self.T = T

        print('q1: {}, q2: {}, T:{}'.format(self.q1, self.q2, self.T))

    @abstractmethod
    def __call__(self, t):
        pass


class LinearShift(BaseShift):
    def __init__(self, q1, q2, T, rng):
        super(LinearShift, self).__init__(q1, q2=q2, T=T, rng=rng)

    def __call__(self, t):
        t += 1
        return self.q1 * (1 - t / self.T) + self.q2 * (t / self.T)


class SquareShift(BaseShift):
    def __init__(self, q1, q2, T, rng, period=None):
        super(SquareShift, self).__init__(q1, q2=q2, T=T, rng=rng)

        if period is None:
            self.period = int(np.sqrt(T))
        else:
            self.period = int(period)
        self.flag = False

    def __call__(self, t):
        if t % self.period == 0:
            self.flag = not self.flag
        return self.q1 if self.flag else self.q2


class SineShift(BaseShift):
    def __init__(self, q1, q2, T, rng, period=None):
        super(SineShift, self).__init__(q1, q2=q2, T=T, rng=rng)

        if period is None:
            self.period = int(np.sqrt(T))
        else:
            self.period = int(period)
        self.alpha_pool = np.sin(np.linspace(0, np.pi, self.period))

    def __call__(self, t):
        alpha = self.alpha_pool[t % self.period]

        return self.q1 * alpha + (1 - alpha) * self.q2


class BernoulliShift(BaseShift):
    def __init__(self, q1, q2, T, rng, prob=None):
        super(BernoulliShift, self).__init__(q1, q2=q2, T=T, rng=rng)
        if prob is None:
            self.prob = 1 / np.sqrt(T)
        else:
            self.prob = prob
        self.q1_flag = True
        self.rng = np.random.default_rng(47)

    def __call__(self, t):
        rand = self.rng.random()
        if rand < self.prob:
            self.q1_flag = not self.q1_flag

        return self.q1 if self.q1_flag else self.q2
