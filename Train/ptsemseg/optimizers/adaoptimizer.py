#!/usr/bin/python
# -*- encoding: utf-8 -*-

import pdb
import torch
import logging

logger = logging.getLogger()

class AdaOptimizer(object):
    def __init__(self,
                model,
                lr0=0.02,
                momentum=0.9,
                wd=1.0e-4,
                warmup_steps=-1000,
                warmup_start_lr=1.0e-5,
                max_iter=150000,
                power=0.9,
                type_='sgd',
                betas=(0.9, 0.999),
                lr_multi=10,
                bn_wd_disable=False,
                *args, **kwargs):

        self.type_ = type_
        self.betas = betas
        self.lr_multi = lr_multi
        self.bn_wd_disable = bn_wd_disable
        self.warmup_steps = warmup_steps
        self.warmup_start_lr = warmup_start_lr
        self.lr0 = lr0
        self.lr = self.lr0
        self.max_iter = float(max_iter)
        self.power = power
        self.it = 0
        wd_params, nowd_params, lr_mul_wd_params, lr_mul_nowd_params = model.get_params()
        if bn_wd_disable:
            param_list = [
                    {'params': wd_params},
                    {'params': nowd_params, 'weight_decay': 0},
                    {'params': lr_mul_wd_params, 'lr_mul': True},
                    {'params': lr_mul_nowd_params, 'weight_decay': 0, 'lr_mul': True}]
        else:
            param_list = [
                    {'params': wd_params},
                    {'params': nowd_params},
                    {'params': lr_mul_wd_params, 'lr_mul': True},
                    {'params': lr_mul_nowd_params, 'lr_mul': True}]

        if self.type_ == 'sgd':
            self.optim = torch.optim.SGD(
                param_list,
                lr = lr0,
                momentum = momentum,
                weight_decay = wd)

        elif self.type_ == 'AdamW':
            self.optim = torch.optim.AdamW(
                param_list,
                lr = lr0,
                betas = tuple(betas),
                weight_decay = wd)

        self.warmup_factor = (self.lr0/self.warmup_start_lr)**(1./self.warmup_steps)


    def get_lr(self):
        if self.it <= self.warmup_steps:
            lr = self.warmup_start_lr*(self.warmup_factor**self.it)
        else:
            factor = (1-(self.it-self.warmup_steps)/(self.max_iter-self.warmup_steps))**self.power
            lr = self.lr0 * factor
        return lr

    def state_dict(self): #now it should works..

        ########### Save the device for each params#################
        #############################################################
        device_list = []
        for state in self.optim.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    device_list.append(v.device)
        ##############################################################

        hyper_param = {'type_':self.type_, 'betas':self.betas, 'warmup_steps': self.warmup_steps, 'warmup_start_lr': self.warmup_start_lr, 'lr0': self.lr0, 'bn_wd_disable': self.bn_wd_disable,
                    'lr': self.lr, 'max_iter':self.max_iter, 'power': self.power, 'lr_multi': self.lr_multi, 'it': self.it, 'sgd': self.optim.state_dict(),
                    'device_list': device_list,'warmup_factor': self.warmup_factor}
        return hyper_param
    
    def load_state_dict(self, ckpt):
        sgd_state_dict = ckpt.pop('sgd', None)

        self.optim.load_state_dict(sgd_state_dict)

        ############### Recover the params to devices #######
        #####################################################
        device_list = ckpt.pop('device_list', None)

        for state in self.optim.state.values():
            for (k, v),dvi in zip(state.items(), device_list):
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(dvi)
        #####################################################


        for key, val in ckpt.items():
            self.__setattr__(key, val)
            
            
    def step(self):
        self.lr = self.get_lr()
        for pg in self.optim.param_groups:
            if pg.get('lr_mul', False):
                pg['lr'] = self.lr * self.lr_multi
            else:
                pg['lr'] = self.lr
        if self.optim.defaults.get('lr_mul', False):
            self.optim.defaults['lr'] = self.lr * self.lr_multi
        else:
            self.optim.defaults['lr'] = self.lr
        self.it += 1
        self.optim.step()
        if self.it == self.warmup_steps+2:
            logger.info('==> warmup done, start to implement poly lr strategy')

    def zero_grad(self):
        self.optim.zero_grad()

