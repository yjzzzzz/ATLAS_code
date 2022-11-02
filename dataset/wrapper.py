# /usr/bin/env python
# -*- coding: utf-8 -*-

from dataset.table_data import get_table_data
from dataset.toy_data import ToyTrainSet, ToyTestSet


def get_dataset(name, cfgs, rng):
    train_set, test_set, info = None, None, {}
    if name == 'toy':
        train_set = ToyTrainSet(cfgs, rng=rng)
        info = train_set.info
        test_set = ToyTestSet(cfgs, info, rng=rng)
    elif name == "table":
        train_set, test_set, info = get_table_data(cfgs)
    else:
        raise NotImplementedError

    return train_set, test_set, info
