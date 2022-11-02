# /usr/bin/env python
# -*- coding: utf-8 -*-

import time


def tensor2scalar(data):
    if data is None:
        return None
    else:
        return data.item()


class Timer(object):
    def __init__(self):
        self.tik()

    def tik(self):
        self.timestamp = time.time()

    def tok(self, desc=''):
        history = self.timestamp
        self.tik()
        print('>>> [{}] Cost time: {:.4f} seconds'.format(desc, self.timestamp - history))
