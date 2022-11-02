# /usr/bin/env python
# -*- coding: utf-8 -*-

from os.path import join

import matplotlib.colors as mcolors
import torch
import torch.nn.functional as F
import torch.utils.data as data

from model.linear import Linear
from utils.shift_simulate import *


def gen_data(centers, cov, priors, rng, sample_num=1000):
    cls_num = len(priors)
    dim = centers.shape[1]
    priors /= np.sum(priors)
    labels = rng.choice(cls_num, size=sample_num, p=priors)
    data = np.zeros((sample_num, dim))
    for i in range(cls_num):
        idx = (labels == i)
        num = idx.sum()
        values = rng.multivariate_normal(centers[i], cov, size=num)
        data[idx] = values

    return data, labels


def normalize(data):
    shape = data.shape

    data = data.view(shape[0], -1)
    min_value = data.min(1, keepdim=True)[0]
    max_value = data.max(1, keepdim=True)[0]
    data = (data - min_value) / (max_value - min_value)
    data = data.view(shape)

    return data


class ToyTrainSet(data.Dataset):
    def __init__(self, cfgs, rng=None):

        if rng is None:
            rng = np.random.default_rng()
        self.rng = rng

        if 'X' in cfgs:
            data = cfgs
        else:
            path = cfgs['source_data']['path']
            data = torch.load(path)

        self.X_bak, self.y = data['X'], data['y']
        self.X = torch.from_numpy(self.X_bak).float()

        self.centers = data['centers']
        self.cov = data['cov']
        self.priors = data['priors']
        self.cls_num = data['cls_num']

        self.info = {
            'dim': self.X.shape[1],
            'cls_num': self.cls_num,
            'centers': self.centers,
            'cov': self.cov,
            'priors': np.array(self.priors),
            'init_priors': np.array(self.priors),
        }

    def __getitem__(self, index):
        return self.X[index], self.y[index]

    def __len__(self):
        return len(self.X)

    def draw(self):
        colors = list(mcolors.TABLEAU_COLORS.keys())
        for i in range(self.cls_num):
            data = self.X_bak[self.y == i]
            plt.scatter(data[:, 0], data[:, 1], c=colors[i], alpha=0.5)
        plt.show()

    def save(self, output):
        torch.save({
            'X': self.X_bak,
            'y': self.y,
            'centers': self.centers,
            'cov': self.cov,
            'priors': self.priors,
            'cls_num': self.cls_num,
        }, output)


class ToyTestSet(data.Dataset):
    def __init__(self, cfgs, info, rng=None):
        if rng is None:
            rng = np.random.default_rng()
        self.rng = rng

        self.info = info
        self.shift_gen = eval(cfgs['shift']['type'])(rng=self.rng,
                                                     **cfgs['shift']['kwargs'])
        self.batch_size = cfgs['online_data']['batch_size']
        self.cfgs = cfgs


    def __getitem__(self, t):
        priors = self.shift_gen(t)
        X, y = gen_data(self.info['centers'],
                        self.info['cov'],
                        priors,
                        self.rng,
                        sample_num=self.batch_size,
                        )
        X = torch.from_numpy(X).float()


        y = torch.from_numpy(y)

        return X, y, priors


def get_toy_data(rng, train_num, test_num, output=None,
                 dim=10, sigma=0.1, cls_num=2, centers=None,
                 source_priors=None, dst_priors=None):

    if dst_priors is None:
        dst_priors = [0.9, 0.1]
    if source_priors is None:
        source_priors = [0.5, 0.5]
    if centers is None:
        centers = rng.random((cls_num, dim))
    print(centers)
    cov = np.identity(dim) * sigma

    train_data, train_labels = gen_data(
        centers=centers,
        cov=cov,
        priors=source_priors,
        rng=rng,
        sample_num=train_num)

    train_cfgs = {
        'X': train_data,
        'y': train_labels,
        'centers': centers,
        'cov': cov,
        'priors': source_priors,
        'normalize': True,
        'cls_num': cls_num,
    }

    test_data, test_labels = gen_data(
        centers=centers,
        cov=cov,
        priors=dst_priors,
        rng=rng,
        sample_num=test_num
    )

    test_cfgs = {
        'X': test_data,
        'y': test_labels,
        'centers': centers,
        'cov': cov,
        'priors': dst_priors,
        'normalize': True,
        'cls_num': cls_num,
    }

    train_set = ToyTrainSet(train_cfgs, rng=rng)
    test_set = ToyTrainSet(test_cfgs, rng=rng)

    if output is not None:
        train_set.save(join(output, 'train_data.pt'))

    info = {
        'dim': dim,
        'cls_num': cls_num,
    }

    return train_set, test_set, info
