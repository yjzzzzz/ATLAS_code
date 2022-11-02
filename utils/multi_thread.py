#!/usr/bin/env python
# -*- coding: utf-8 -*-

from concurrent import futures
from time import time


def monotonic(idxs):
    return all([idxs[i] < idxs[i + 1] for i in range(len(idxs) - 1)])


class Pipeline(object):
    def __init__(self, *funcs):
        self.funcs = funcs

    def flow(self, input, idx=0):
        output = input
        for func in self.funcs:
            output = func(output)

        return output, idx

    def add_func(self, func):
        self.funcs += func


class MultiThreadHelper(object):
    # commands:
    def __init__(self, commands, threads=1, prefix="", multi_process=True):
        self.threads = threads
        self.commands = commands
        self.prefix = prefix
        self.results = []
        if multi_process:
            self.executor = futures.ProcessPoolExecutor
        else:
            self.executor = futures.ThreadPoolExecutor

    def run(self):
        if self.threads > 0:
            with self.executor(max_workers=self.threads) as executor:
                fs = [executor.submit(*cmd) for cmd in self.commands]

                for f in futures.as_completed(fs):
                    try:
                        self.results.append(f.result())
                    except Exception as e:
                        raise e
        else:
            # for task in tqdm(self.commands, desc=self.prefix):
            for task in self.commands:
                func = task[0]
                args = task[1:]
                result = func(*args)
                self.results.append(result)

        # self.results.sort(key=lambda result: result[1])

    def __call__(self):
        start = time()
        self.run()
        dur = time() - start
        # print(self.prefix + ' finished. Totally %s seconds cost.' % dur)
        return self.results
