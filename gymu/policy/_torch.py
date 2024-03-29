#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
 Created on 14-01-2021 12:14:17

 [Description]
"""
__author__ ="Benedict Wilkins"
__email__ = "benrjw@gmail.com"
__status__ ="Development"

import torch
from . import _base

class NeuralPolicy(_base.DiscretePolicy): # TODO

    def __init__(self, action_space):
        super(NeuralPolicy, self).__init__(action_space)

    def sample(self, *args, **kwargs):
        v = self.nn(*args, **kwargs)
        p = self.p(v)
        return torch.multinomial(p, 1).squeeze()