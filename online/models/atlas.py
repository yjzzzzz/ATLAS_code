# /usr/bin/env python
# -*- coding: utf-8 -*-

from collections import OrderedDict, defaultdict

import torch
from prettytable import PrettyTable

from online.models.meta import Hedge
from online.models.ogd import Base, CustomOGD, OptOGD
from online.utils.lr import SelfTuningLr, OptSelfTuningLr
from online.utils.schedule import DiscretizedSSP, Schedule
from utils.tools import Timer


class ATLAS(Base):
    def __init__(self, cfgs=None, seed=None, **alg_kwargs):
        super(ATLAS, self).__init__(cfgs=cfgs, seed=seed, **alg_kwargs)
        if cfgs is None:
            cfgs = {}
        ssp = DiscretizedSSP(min_step=alg_kwargs['min_step'],
                             max_step=alg_kwargs['max_step'],
                             grid=cfgs.get('grid', 2))
        print(ssp)
        self.ssp = ssp
        self._schedule = self.expert_steup(ssp, **alg_kwargs)

        self._meta = self.meta_setup(ssp, cfgs)

        self.loss_vector = torch.zeros(len(ssp))

        self._t = 0
        self.combine = cfgs.get('combine', 'weight')
        self.print = cfgs.get('pretty_print', False)

    def expert_steup(self, ssp, **alg_kwargs):
        return Schedule(ssp, CustomOGD,
                        ['model'],
                        use_optimism=False,
                        thread=self.cfgs.get('thread', 0),
                        **alg_kwargs)

    def meta_setup(self, ssp, cfgs):
        G = cfgs['G']
        D = cfgs['D']
        lr_scale = cfgs.get('lr_scale', 1.0)

        lr = SelfTuningLr(scale=lr_scale / (G * D))
        meta = Hedge(N=len(ssp), lr=lr, prior=cfgs.get("meta_init", 'uniform'))
        print('Meta init: ' + cfgs.get("meta_init", 'uniform'))
        print('Meta LR: {}'.format(lr_scale / (G * D)))

        return meta

    @torch.no_grad()
    def predict(self, data):
        prob = self._meta.get_prob()
        if self.combine == 'weight':
            output = self.model(data)
            pred = output.argmax(-1)
        elif self.combine == "output":
            soft_out = []
            data = data.to(self.device).to(torch.float32)

            for model in self._schedule.bases:
                output = model.model(data)
                soft_out.append(output.softmax(-1))
            soft_out = torch.stack(soft_out, dim=0)
            num_bases = len(self._schedule)
            combine_out = torch.mm(prob.view(1, num_bases), soft_out.view(num_bases, -1))
            combine_out = combine_out.reshape(soft_out.shape[1:])
            pred = combine_out.argmax(-1)
        else:
            return NotImplementedError

        self.pretty_print(prob, 'Weight')
        return pred

    def set_func(self, func):
        super().set_func(func)
        self._schedule.set_func(func)

    def get_meta_lr(self):
        return self._meta.lr

    def get_meta_prob(self):
        return self._meta.get_prob()

    def opt(self, target_data):
        self._t += 1

        # optimization
        self._meta.update_lr()
        self._meta.opt(self.loss_vector)  # Meta

        loss_vector = torch.zeros(len(self._schedule))  # Base
        for batch_idx, (data, target) in enumerate(self.cache):
            loss_vector += self._schedule.opt(data, target, target_data)

        self.loss_vector = loss_vector / len(self.dataloader)

    def eval(self):
        estimate_loss, underlying_loss = 0, 0

        for batch_idx, (data, target) in enumerate(self.cache):
            output = self.model(data)
            info = self.criterion(output.float(), target)
            estimate_loss += info['estimate'].item()
            if info['underlying'] is None:
                underlying_loss = None
            else:
                underlying_loss += info['underlying'].item()

        estimate_loss /= len(self.dataloader)
        if underlying_loss is not None:
            underlying_loss /= len(self.dataloader)

        return estimate_loss, underlying_loss

    def load_model(self):
        def weight_combine(x_bases, prob):
            prob = prob.to(x_bases.device)

            x_shape = x_bases.shape
            num_bases = x_shape[0]

            x_combined = torch.mm(prob.view(1, num_bases), x_bases.view(num_bases, -1))
            x_combined = x_combined.reshape(x_shape[1:])

            return x_combined

        prob = self._meta.get_prob()
        weights = self._schedule.get_x()
        combine_weights = OrderedDict()
        names = weights[0].keys()
        for name in names:
            bases = [base[name] for base in weights]
            combine_weights[name] = weight_combine(torch.stack(bases), prob)

        self.model.load_state_dict(combine_weights)

    def forward(self, target_data, prior_estimate, t):

        # evaluate
        self.load_model()
        pred = self.predict(target_data)
        estimate_loss, underlying_loss = self.eval()

        # update
        self.opt(target_data)

        return pred, estimate_loss, underlying_loss

    def pretty_print(self, values, name):
        if self.print:
            table = PrettyTable()
            steps = ['{:.8f}'.format(step) for step in self.ssp._step_pool[:len(values)]]
            table.field_names = ['Lr'] + steps
            values = values.cpu().detach().numpy()
            values = ['{:.2f}'.format(v) for v in values]

            table.add_row(['Val'] + values)
            print('>>>{}'.format(name))
            print(table)


class ATLASADA(ATLAS):
    def __init__(self, cfgs=None, seed=None, **alg_kwargs):
        super(ATLASADA, self).__init__(cfgs, seed, **alg_kwargs)

        self.opt_vector = torch.zeros(len(self._schedule))
        self.optimism = alg_kwargs['optimism']

        self.save_loss = cfgs.get('save_loss', False)
        print("Save Loss and Weight: {}".format(self.save_loss))
        if self.save_loss:
            self.record = defaultdict(list)

    def expert_steup(self, ssp, **alg_kwargs):
        return Schedule(ssp, OptOGD,
                        ['model'],
                        use_optimism=True,
                        thread=self.cfgs.get('thread', 0),
                        **alg_kwargs)

    def meta_setup(self, ssp, cfgs):
        opt_tuning = cfgs.get('meta_opt_lr', False)
        if opt_tuning:
            lr = OptSelfTuningLr(N=len(ssp))
            meta = Hedge(N=len(ssp), lr=lr, prior=cfgs.get("meta_init", 'uniform'))
            print('Meta init: ' + cfgs.get("meta_init", 'uniform'))
            print('Meta LR: {}'.format(lr))
            return meta
        else:
            return super().meta_setup(ssp, cfgs)

    def opt(self, target_data):
        self._t += 1

        self._meta.opt(self.loss_vector)  # Meta

        # for batch_idx, (data, target) in enumerate(self.dataloader):
        for batch_idx, (data, target) in enumerate(self.cache):
            self._schedule.opt(data, target, target_data)

    def optimism_opt(self, target_data, prior_estimate, t):
        timer = Timer()
        self._meta.update_lr(
            opt=self.opt_vector,
            loss=self.loss_vector
        )

        self.optimism.set_info(target_data, prior_estimate, t)

        # optimism opt
        opt_vector = torch.zeros(len(self._schedule))
        loss_vector = torch.zeros(len(self._schedule))
        # for batch_idx, (data, target) in enumerate(self.dataloader):
        for batch_idx, (data, target) in enumerate(self.cache):
            opt, loss = self._schedule.optimism_opt(data, target)
            opt_vector += opt
            loss_vector += loss

        self.loss_vector = loss_vector / len(self.dataloader)
        self.opt_vector = opt_vector / len(self.dataloader)

        self._meta.optimism_opt(self.opt_vector)
        self.pretty_print(self.loss_vector, 'Loss')

        if self.save_loss:
            self.record['loss'].append(self.loss_vector)
            self.record['opt'].append(self.opt_vector)
            self.record['weight'].append(self._meta.get_prob())

    def forward(self, target_data, prior_estimate, t):

        self.optimism_opt(target_data, prior_estimate, t)

        return super().forward(target_data, prior_estimate, t)
