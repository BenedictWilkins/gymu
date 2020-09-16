#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
    Created on 16-09-2020 13:21:11

    [Description]
"""
__author__ = "Benedict Wilkins"
__email__ = "benrjw@gmail.com"
__status__ = "Development"

from collections import namedtuple

import numpy as np

class mode:
    
    def __init__(self, *data):
        self.__data = data
    
    def __getitem__(self, index):
        return self.__data[index]
    
    def __iter__(self):
        return self.__data.__iter__()

    def __str__(self):
        return "mode-{0}".format(self.__class__.__name__)

    def __repr__(self):
        return str(self)

class s(mode):
    
    def __init__(self, state=None, **kwargs):
        super(s, self).__init__(state)
    
    @property
    def state(self):
        return self[0]
        
class r(mode):
    
    def __init__(self, reward=None, **kwargs):
        super(r, self).__init__(reward)

    @property
    def reward(self):
        return self[0]

class sa(mode):
    
    def __init__(self, state=None, action=None, **kwargs):
        super(sa, self).__init__(state, action)

    @property
    def state(self):
        return self[0]

    @property
    def action(self):
        return self[1]

class ss(mode):
    
    def __init__(self, state=None, nstate=None, **kwargs):
        super(ss, self).__init__(state, nstate)

    @property
    def state(self):
        return self[0] 

    @property
    def nstate(self):
        return self[1]


class sr(mode):
    
    def __init__(self, state=None, reward=None, **kwargs):
        super(sr, self).__init__(state, reward)

    @property
    def state(self):
        return self[0] 

    @property
    def reward(self):
        return self[1]

        
class sar(mode):
    
    def __init__(self, state=None, action=None, reward=None, **kwargs):
        super(sar, self).__init__(state, action, reward)

    @property
    def state(self):
        return self[0] 

    @property
    def action(self):
        return self[1]

    @property
    def reward(self):
        return self[2] 

class ars(mode):
    
    def __init__(self, action=None, reward=None, nstate=None, **kwargs):
        super(ars, self).__init__(action, reward, nstate)

    @property
    def action(self):
        return self[0]

    @property
    def reward(self):
        return self[1] 

    @property
    def nstate(self):
        return self[2] 

class sas(mode):
        
    def __init__(self, state=None, action=None, nstate=None, **kwargs):
        super(sas, self).__init__(state, action, nstate)
    
    @property
    def state(self):
        return self[0] 

    @property
    def action(self):
        return self[1]

    @property
    def nstate(self):
        return self[2] 
        
class sars(mode):
        
    def __init__(self, state=None, action=None, reward=None, nstate=None, **kwargs):
        super(sars, self).__init__(state, action, reward, nstate)

    @property
    def state(self):
        return self[0] 

    @property
    def action(self):
        return self[1]

    @property
    def reward(self):
        return self[2]

    @property
    def nstate(self):
        return self[3] 

def pack(modes):
    """ 
        Packs a list of modes into numpy arrays
    """
    return tuple([np.array(d) for d in [i for i in zip(*modes)]])
