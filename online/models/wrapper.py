# /usr/bin/env python
# -*- coding: utf-8 -*-

from online.models.atlas import ATLAS, ATLASADA
from online.models.ogd import *
from online.utils.risk import RewritingRisk
from online.utils.hint_function import *

def get_model(cfgs, min_step, max_step, model, train_set, device, init, rng, info):
    optimism = None
    if 'optimism' in cfgs['kwargs']:
        loss_func = RewritingRisk(device=device,
                                  nn_loss=cfgs['kwargs'].get('nn_loss', False))
        kwargs = cfgs['kwargs']['optimism'].get('kwargs', {})
        kwargs['rng'] = rng
        optimism = eval(cfgs['kwargs']['optimism']['type'])(loss_func=loss_func,
                                                            dim=info['dim'],
                                                            cls_num=info['cls_num'],
                                                            **kwargs)

    alg_kwargs = {
        'model': model,
        'dataset': train_set,
        'device': device,
        'batch_size': cfgs['kwargs']['source_batch_size'],
        'init': init,
    }
    if cfgs['algorithm'] == 'OGD':
        alg_kwargs.update({
            'stepsize': cfgs['kwargs'].get("lr", min_step),
            'projection': cfgs['kwargs'].get('projection', False),
        })
        online_alg = CustomOGD(cfgs=cfgs['kwargs'], **alg_kwargs)
    elif cfgs['algorithm'] == 'ATLAS':
        alg_kwargs.update({
            'min_step': min_step,
            'max_step': max_step,
            'projection': cfgs['kwargs'].get('projection', False),
        })
        online_alg = ATLAS(cfgs=cfgs['kwargs'], **alg_kwargs)
    elif cfgs['algorithm'] == 'ATLASADA':
        alg_kwargs.update({
            'min_step': min_step,
            'max_step': max_step,
            'projection': cfgs['kwargs'].get('projection', False),
            'optimism': optimism,
            'opt_lr': cfgs["kwargs"].get("opt_lr", 1),
            'opt_epoch': cfgs["kwargs"].get("opt_epoch", 7),
            'opt_optimizer': cfgs["kwargs"].get("opt_optimizer", 'LBFGS'),
        })
        online_alg = ATLASADA(cfgs=cfgs['kwargs'], **alg_kwargs)
    elif cfgs['algorithm'] == 'FIX':
        online_alg = Base(**alg_kwargs)
    else:
        raise NotImplementedError

    return online_alg
