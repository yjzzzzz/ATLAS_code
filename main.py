# /usr/bin/env python
# -*- coding: utf-8 -*-

import json as js
import os
import time
from os.path import join

import numpy as np
import torch.utils.data as data
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from dataset.wrapper import get_dataset
from model.wrapper import get_cls_model
from online.models.wrapper import get_model
from online.utils.bbse import BBSE
from online.utils.risk import *
from utils.argparser import argparser

from utils.tools import Timer



def write(writer, info, t):
    for k, v in info.items():
        writer.add_scalar(k, v, t)


def set_cpu_num(cpu_num=8):
    os.environ['OMP_NUM_THREADS'] = str(cpu_num)
    os.environ['OPENBLAS_NUM_THREADS'] = str(cpu_num)
    os.environ['MKL_NUM_THREADS'] = str(cpu_num)
    os.environ['VECLIB_MAXIMUM_THREADS'] = str(cpu_num)
    os.environ['NUMEXPR_NUM_THREADS'] = str(cpu_num)
    torch.set_num_threads(cpu_num)
    torch.set_num_interop_threads(cpu_num)


def run(T, test_set, estimator, model, cfgs, info, device='cuda', writer=None):
    record = []
    cumulative_estimate_loss, cumulative_underlying_loss, error_cnt = 0, 0, 0

    loss_cfgs = cfgs['Online']['kwargs'].get('loss', {})
    loss_name = loss_cfgs.get('name', 'RewritingRisk')
    nn_loss = loss_cfgs.get('nn_loss', False) or \
              cfgs['Online']['kwargs'].get('nn_loss', False)
    loss_func = eval(loss_name)(device=device, nn_loss=nn_loss)
    print('Loss {}, Use nn: {}'.format(loss_name, nn_loss))

    time_helper = Timer()
    time_helper.tik()
    for t in tqdm(range(T)):

        target_data, target_label, underlying_priors = test_set[t]

        target_data = target_data.to(device)
        sample_num = target_data.shape[0]
        prior_estimate = estimator.estimate(target_data).to(device)
        loss_func.set_priors(prior_estimate, underlying_priors)
        model.set_func(loss_func)

        result = model.forward(target_data, prior_estimate, t)
        if len(result) == 3:
            pred, estimate_loss, underlying_loss = result
            loss_info = None
        else:
            pred, estimate_loss, underlying_loss, loss_info = result

        ori_pred = None
        if isinstance(pred, dict):
            ori_pred = pred['Ori']
            pred = pred['Opt']

        target_label = target_label.to(pred.device)
        _error_cnt = (pred.view_as(target_label) != target_label).sum().item()
        error_cnt += _error_cnt

        avg_error = error_cnt / ((t + 1) * sample_num)

        cumulative_estimate_loss += estimate_loss

        if underlying_loss is not None:
            cumulative_underlying_loss += underlying_loss
        if writer is not None:
            res_info = {
                'Estimate/1-Estimate Loss': estimate_loss,
                'Estimate/2-Cumulative Loss': cumulative_estimate_loss,
                'Estimate/3-Prior[0]': prior_estimate[0],
                'Underlying/3-Avg Error': avg_error,
                'Underlying/4-Cumulative Error': error_cnt,
            }
            if underlying_loss is not None:
                res_info.update({
                    'Underlying/1-Underlying Loss': underlying_loss,
                    'Underlying/2-Cumulative Loss': cumulative_underlying_loss,
                    'Underlying/5-Prior[0]': underlying_priors[0],
                })

            write(writer, res_info, t)
            for k, v in res_info.items():
                res_info[k] = v.item() if isinstance(v, torch.Tensor) else v
            record.append(res_info)

        if t % cfgs.get('log_interval', 100) == 0:
            time_helper.tok('{} rounds'.format(t))
            print(
                '\n[Time {}] Estimate Loss: {}, Underlying Loss: {}, Avg Error: {}'.format(t, estimate_loss,
                                                                                           underlying_loss,
                                                                                           avg_error))
    return record


def stepsize(cfgs, sigma_min, K):
    alg = cfgs['algorithm']
    cfgs = cfgs['kwargs']
    D = float(cfgs['D'])
    G = float(cfgs['G'])
    T = cfgs['T']

    print('D: {}, G: {}, T:{}'.format(D, G, T))
    print('K: {}, Sigma: {}'.format(K, sigma_min))

    if alg == 'ATLASADA':
        min_step = (D * sigma_min) / (2 * G * ((K * T) ** 0.5))
        max_step = D * ((1 + 2 * T) ** 0.5)
    else:
        min_step, max_step = D / (G * (T ** 0.5)), D / G

    max_step_clip = cfgs.get('max_step_clip', max_step)
    max_step = min(max_step, max_step_clip)

    return min_step, max_step


if __name__ == "__main__":
    cfgs = argparser()

    device = cfgs.get('device', 'cpu')
    cpu_num = cfgs.get('cpu_num', 8)
    set_cpu_num(cpu_num)
    rng = np.random.default_rng(cfgs['random_seed'])

    train_set, test_set, info = get_dataset(
        name=cfgs['Data']['name'],
        cfgs=cfgs['Data']['kwargs'],
        rng=rng
    )

    # BBSE
    bbox, _ = get_cls_model(cfgs, info, device)

    source_loader = data.DataLoader(train_set,
                                    batch_size=cfgs['BlackBox']['source_batch_size'],
                                    shuffle=True,
                                    pin_memory=False)

    estimator = BBSE(bbox=bbox,
                     source_loader=source_loader,
                     cls_num=info['cls_num'],
                     device=device,
                     kwargs=cfgs['BlackBox'].get("kwargs", {}),
                     )
    sigma_min = estimator.get_sigma_min()

    min_step, max_step = stepsize(cfgs['Online'],
                                  sigma_min,
                                  info['cls_num'])
    model, init = get_cls_model(cfgs, info, device)
    online_alg = get_model(cfgs['Online'], min_step, max_step, model, train_set, device, init, rng, info)

    writer = None
    if cfgs['Online'].get('write', True):
        writer = SummaryWriter(join(cfgs['output'],
                                    'runs_{}'.format(time.strftime("%a_%b_%d_%H:%M:%S",
                                                                   time.localtime()))))
    record = run(cfgs['round'], test_set, estimator,
                                online_alg, cfgs, info,
                                device=device, writer=writer)

    with open(join(cfgs["output"], 'result.json'), 'w') as fw:
        js.dump(record, fw, indent=4)


