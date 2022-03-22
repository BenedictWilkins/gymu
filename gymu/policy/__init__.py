#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
 Created on 14-01-2021 12:11:59

 [Description]
"""
__author__ ="Benedict Wilkins"
__email__ = "benrjw@gmail.com"
__status__ ="Development"

from ._base import *

class Uniform(DiscretePolicy):

    def __init__(self, env, *args, **kwargs):
        super(Uniform, self).__init__(env.action_space, *args, **kwargs)

    def sample(self, *args, **kwargs):
        return self.action_space.sample()
    
    
    