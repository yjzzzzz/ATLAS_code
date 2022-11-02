# /usr/bin/env python
# -*- coding: utf-8 -*-

from os.path import isdir, join

import pandas as pd
import torch
import torch.utils.data as data
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

from utils.shift_simulate import *


class CustomDataset(data.Dataset):
    def __init__(self, data, labels, datetime, ori_labels=None):
        self.data = data
        self.labels = labels
        self.datetime = datetime
        self.ori_labels = ori_labels

        if isinstance(self.data, np.ndarray):
            self.data = torch.from_numpy(self.data)
        self.data = self.data.float()

    def __getitem__(self, index):
        pass

    def __len__(self):
        return len(self.data)


class TrainDataset(CustomDataset):
    def __init__(self, data, labels, datetime, ori_labels=None):
        super(TrainDataset, self).__init__(data, labels, datetime, ori_labels)

    def __getitem__(self, index):
        return self.data[index], self.labels[index]

    def fission(self, test_ratio):
        train_data, test_data, \
        train_labels, test_labels, \
        train_datetime, test_datetime = \
            train_test_split(
                self.data, self.labels, self.datetime,
                test_size=test_ratio,
                shuffle=True,
                random_state=1024
            )

        return self.__class__(train_data, train_labels, test_datetime), \
               self.__class__(test_data, test_labels, test_datetime)


class TestDataset(CustomDataset):
    def __init__(self, data, labels, datetime, **kwargs):
        super(TestDataset, self).__init__(data, labels, datetime)
        self.rng = np.random.default_rng(kwargs.get("seed", 1214))
        self.interval = kwargs.get("interval", 10)
        self.ptr = 0
        print('Online Batch Size: {}'.format(self.interval))
        if isinstance(self.labels, np.ndarray):
            self.labels = torch.from_numpy(self.labels)

        self.drop = kwargs.get('drop', False)

    def __getitem__(self, t):
        X = self.data[self.ptr:self.ptr + self.interval]
        y = self.labels[self.ptr:self.ptr + self.interval]
        self.ptr += self.interval

        if self.drop:
            idx = self.rng.integers(self.interval)
            X = torch.cat((X[:idx], X[idx + 1:]), dim=0)
            y = torch.cat([y[:idx], y[idx + 1:]], dim=0)

        return X, y, None


class SimulateTestDataset(CustomDataset):
    def __init__(self, data, labels, datetime, **kwargs):
        super(SimulateTestDataset, self).__init__(data, labels, datetime)
        self.rng = np.random.default_rng(kwargs.get("seed", 1214))
        self.shift_gen = eval(kwargs['shift']['type'])(rng=self.rng,
                                                       **kwargs['shift']['kwargs'])
        self.batch_size = kwargs['batch_size']

        self.data_menu = {}
        for i in np.unique(self.labels):
            self.data_menu[i] = np.where(self.labels == i)[0]

    def __getitem__(self, t):
        priors = self.shift_gen(t)
        cls_num = len(priors)
        dim = self.data.shape[1]
        priors /= np.sum(priors)
        labels = self.rng.choice(cls_num, size=self.batch_size, p=priors)
        data = torch.zeros((self.batch_size, dim), device=self.data.device)

        for i in range(cls_num):
            idx = (labels == i)
            num = idx.sum()
            data_idx_idx = self.rng.choice(len(self.data_menu[i]), size=num, replace=False)

            data_idx = self.data_menu[i][data_idx_idx]
            data[idx] = self.data[data_idx]

        return data, torch.from_numpy(labels), priors


def get_table_data(cfgs, bbox=False):
    path = cfgs['source_data']['path']
    if isdir(path):
        train_data = pd.read_csv(join(path, 'train_data.csv'))
        test_data = pd.read_csv(join(path, 'test_data.csv'))
        features_train = train_data.drop(columns=['y', 'date']).values
        features_test = test_data.drop(columns=["y", "date"]).values
        labels_train, labels_test = train_data['y'], test_data['y']
        dates_train, dates_test = train_data['date'], test_data['date']

        if cfgs['online_data'].get('shuffle', False):
            features_test, labels_test, dates_test = \
                shuffle(features_test, labels_test, dates_test, random_state=cfgs['seed'])

    else:
        data = pd.read_csv(path)
        data = data.sort_values(by='date')

        features = data.drop(columns=['y', 'date']).values
        labels = data['y'].values
        dates = data['date'].values

        total_num = len(labels)
        split_ratio = cfgs.get("train_ratio", 0.5)
        print('Split ratio: {}'.format(split_ratio))
        train_num = int(total_num * split_ratio)
        features_train, features_test = features[:train_num], features[train_num:]
        labels_train, labels_test = labels[:train_num], labels[train_num:]
        dates_train, dates_test = dates[:train_num], dates[train_num:]

    le = preprocessing.LabelEncoder()
    _labels_train = le.fit_transform(labels_train)
    _labels_test = le.transform(labels_test)

    train_set = TrainDataset(
        data=features_train,
        ori_labels=labels_train,
        labels=_labels_train,
        datetime=dates_train,
    )

    if bbox:
        test_set = TrainDataset(
            data=features_test,
            ori_labels=labels_train,
            labels=_labels_test,
            datetime=dates_test,
        )
    else:
        dataset_type = cfgs['online_data'].get('type', 'TestDataset')
        test_set = eval(dataset_type)(
            data=features_test,
            labels=_labels_test,
            datetime=dates_test,
            seed=cfgs['seed'],
            **cfgs['online_data'],
        )

    cls_num = len(np.unique(_labels_train))
    count = torch.bincount(torch.from_numpy(_labels_train), minlength=cls_num)
    priors = count / count.sum()

    info = {
        'cls_num': cls_num,
        'dim': features_train.shape[1],
        'init_priors': priors,
    }

    return train_set, test_set, info

