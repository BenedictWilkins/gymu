#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
    Created on 16-09-2020 13:21:11

    [Description]
"""
__author__ = "Benedict Wilkins"
__email__ = "benrjw@gmail.com"
__status__ = "Development"

import numpy as np

__all__ = ['pack', 'mode']

def pack(modes):
    """ Pack a list of modes into numpy arrays. """
    modes = list(modes)
    mode = type(modes[0])
    return mode(*[np.array(d) for d in [i for i in zip(*modes)]])

def mode(m):
    return _meta_mode._cast_from(m)

class _meta_mode(type):
    
    MODE_INDEX = dict(   
        # INDEX is based on the returned sequence of the gymu iterator step, which is, in order: state, action, next_state, reward, done, info
        s    = [0],            # state
        sr   = [0,3],          # state, reward
        sa   = [0,1],          # state, action
        ss   = [0,2],          # state, next_state
        sar  = [0,1,3],        # state, action, reward
        sas  = [0,1,2],        # state, action next_state
        sars = [0,1,3,2],      # state, action, reward, next_state
    )
    MODE_INDEX.update({k + "d":v + [4] for k,v in MODE_INDEX.items()})
    MODE_INDEX.update({k + "i":v + [5] for k,v in MODE_INDEX.items()}) 
    
    PROPERTIES = ['state', 'action', 'reward', 'next_state', 'done', 'info']
    PROPERTIES_ALIAS = ['s', 'a', 'r', 's', 'd', 'i']

    def __new__(cls, name, bases, dct):
        if name != '_mode':
            prop_map = dict(s='state', a='action', r='reward', d='done', i='info')
            # add properties
            for i, n in enumerate(name):
                dct[prop_map[n]] = property(lambda self, j=i: self._data[j])
                if n == 's': 
                    prop_map['s'] = 'next_state'
            # TODO create a suitable __init__ function for each subclass... 
            
            m = super().__new__(cls, name, bases, dct)
            m.__index__ = _meta_mode.MODE_INDEX.get(name, None)
            
        else:
            m = super().__new__(cls, name, bases, dct)
        return m

    def _cast_from(m):
        c = None
        subclasses = {c.__name__:c for c in _mode.__subclasses__()}
        if isinstance(m, str):
            c = subclasses.get(m, None)
        elif hasattr(m, "__iter__"):
            try:
                indx = [_meta_mode.PROPERTIES.index(i) for i in m]
            except:
                raise TypeError(f"Cannot cast {m} to a valid mode, given values must be in {_meta_mode.PROPERTIES}")
            c_str = "".join(_meta_mode.PROPERTIES_ALIAS[i] for i in indx)
            c = subclasses.get(c_str, None)
        if c is not None:
            return c
        raise TypeError(f"Cannot cast {m} to a valid mode.")
    
class _mode(metaclass=_meta_mode):

    def __init__(self, *data):
        self._data = data
    
    def __getitem__(self, index):
        return self._data[index]
    
    def __iter__(self):
        return self._data.__iter__()

    def __str__(self):
        return "mode-{0}".format(self.__class__.__name__)

    def __repr__(self):
        return str(self)
    
    def __len__(self):
        return len(self._data)

    def tuple(self): # convert to a python tuple
        return self._data

# make mode subclasses
for m, i in _meta_mode.MODE_INDEX.items():
    cls = type(m, (_mode,), {})
    globals()[cls.__name__] = cls
    __all__.append(cls.__name__)

