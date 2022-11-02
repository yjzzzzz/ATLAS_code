# /usr/bin/env python
# -*- coding: utf-8 -*-

import cvxpy as cp
import numpy as np
import torch
import torch.nn.functional as F


class BBSE(object):
    def __init__(self, bbox, source_loader, cls_num, device, kwargs):
        bbox.eval()
        self.bbox = bbox
        self.cls_num = cls_num
        self.device = device
        self.cm, self.cm_inv, self.sigma_min = self.confusion_matrix(source_loader)

        self.nn_clip = kwargs.get("nn_clip", False)
        self.softmax = kwargs.get("softmax", False)

        self.use_history = kwargs.get('use_history', False)
        self.smooth_ratio = kwargs.get("smooth_ratio", 0)
        self.history = None

        self.use_reg = kwargs.get('use_reg', False)
        self.lamda = kwargs.get('lambda', 0)

    def get_sigma_min(self):
        return self.sigma_min

    @torch.no_grad()
    def confusion_matrix(self, source_loader):
        cmatrix = torch.zeros((self.cls_num, self.cls_num), device=self.device)
        for batch_idx, (data, target) in enumerate(source_loader):
            data, target = data.to(self.device), target.to(self.device)
            output = self.bbox(data)
            pred = output.argmax(-1)
            indices = self.cls_num * pred + target

            m = torch.bincount(indices, minlength=self.cls_num ** 2)
            cmatrix += m.reshape(cmatrix.shape)

        cmatrix_joint = F.normalize(cmatrix.float().view(-1),
                                    p=1, dim=0).reshape(self.cls_num, self.cls_num)
        u, s, v = torch.svd(cmatrix_joint)

        cmatrix = F.normalize(cmatrix.float(), p=1, dim=0)
        _u, _s, _v = torch.svd(cmatrix)

        cm_inv = cmatrix.inverse()

        print("[BBSE] Joint Minimum Singular Value: {}".format(s.min().item()))
        print("[BBSE] Conditional Minimum Singular Value: {}".format(_s.min().item()))

        return cmatrix, cm_inv, _s.min().item()

    @torch.no_grad()
    def estimate(self, target_x):

        output = self.bbox(target_x.to(torch.float32))
        pred = output.argmax(-1)
        q_count = torch.bincount(pred, minlength=self.cls_num)
        q_f = F.normalize(q_count.float(), p=1, dim=0)

        if self.use_reg:
            q_estimate = self.reg_estimate(q_f)
        else:
            q_estimate = self.cm_inv.matmul(q_f.T)

            if self.nn_clip:
                q_estimate = q_estimate.clamp(min=0)

            if self.softmax:
                q_estimate = q_estimate.softmax(dim=-1)

            if self.use_history and self.history is not None:
                q_estimate = self.history * self.smooth_ratio + q_estimate * (1 - self.smooth_ratio)

        self.history = q_estimate
        return q_estimate

    def reg_estimate(self, q_f):
        C = self.cm.detach().numpy()
        q = q_f.detach().numpy().flatten()
        ones = np.ones(len(q))

        history = None
        if self.history is not None:
            history = self.history.detach().numpy().flatten()

        ans = cp.Variable(len(q))
        loss_func = cp.pnorm(C @ ans - q, p=2)
        if history is not None:
            loss_func += self.lamda * cp.pnorm(ans / history - ones, p=2)
        objective = cp.Minimize(loss_func)
        prob = cp.Problem(objective)

        prob.solve()

        q_estimate = torch.from_numpy(ans.value).float()

        return q_estimate
