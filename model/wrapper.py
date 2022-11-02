# /usr/bin/env python
# -*- coding: utf-8 -*-

import torch

from model.linear import Linear

def get_cls_model(cfgs, info, device):
    if 'path' in cfgs['BlackBox']:
        init = torch.load(cfgs['BlackBox']['path'], map_location='cpu')
    else:
        init = None

    print('Init: {}'.format(init is not None))

    model_name = cfgs['BlackBox'].get('type', 'Linear')
    print('BBox model: {}'.format(model_name))

    if model_name == 'Linear':
        model = Linear(
            input_dim=info['dim'],
            output_dim=info['cls_num'],
            R=cfgs['Online']['kwargs']['D'] / 2
        )
    else:
        raise NotImplementedError

    if init is not None:
        model.load_state_dict(init)

    return model.to(device), init
