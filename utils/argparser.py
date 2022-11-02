# /usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
from os.path import join, exists

import yaml


def argparser():
    parser = argparse.ArgumentParser()

    parser.add_argument("--expr_space",
                        type=str,
                        help='the path of the experiment')
    args = parser.parse_args()

    cfg_path = join(args.expr_space, 'config.yaml')
    if not exists(cfg_path):
        raise ValueError
    with open(cfg_path) as f:
        cfg = yaml.load(f, Loader=yaml.Loader)

    cfg['output'] = join(args.expr_space, 'output')

    return cfg
