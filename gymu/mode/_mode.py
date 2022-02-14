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
    return _cast_from(m)

class _mode(tuple):

    def __new__(cls, *data, **kwdata):
        if len(data) > 0 and len(kwdata) > 0:
            raise ValueError("Mode arguments cannot be combined, specify either data or kwdata.")
        
        keys = tuple([MODE_PROPERTIES[i] for i in cls.__index__])
        if len(kwdata) > 0: # construct from dictionary
            if len(keys) != len(kwdata.keys()) or not all([x in keys for x in kwdata.keys()]):
                raise ValueError(f"Invalid keys {list(kwdata.keys())} specified for mode {cls}, must match {list(keys)}.")
            data = [x[1] for x in sorted(kwdata.items(), key = lambda x: MODE_PROPERTIES.index(x[0]))]

        if len(data) != len(cls.__index__):
            if len(data) > 0 and isinstance(data[0], (list, tuple, dict)): 
                raise ValueError(f"Invalid argument type {type(data[0])} did you forget to unpack with * or ** ?")
            raise ValueError(f"Requires {len(cls.__index__)} arguments  to match {keys}, only {len(data)} were supplied.")
        return tuple.__new__(cls, data)

    def __str__(self):
        return "mode-{0}".format(self.__class__.__name__)

    def __repr__(self):
        return str(self)

    def items(self):
        return zip(self.keys(), self.values())
    
    def values(self):
        return tuple([x for x in self])

    def keys(self):
        return tuple([MODE_PROPERTIES[i] for i in self.__index__])

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

MODE_PROPERTIES = ('state', 'action', 'reward', 'next_state', 'done', 'info')
MODE_PROPERTIES_ALIAS = ('s', 'a', 'r', 's', 'd', 'i')

def _make_mode_class(name, index):
    prop_map = dict(s='state', a='action', r='reward', d='done', i='info')
    properties = {}
    for i, n in enumerate(name):
        properties[prop_map[n]] = property(lambda self, j=i: self[j])
        if n == 's': 
            prop_map['s'] = 'next_state'
    # TODO create a suitable __init__ function for each subclass... ? 
    cls = type(m, (_mode,), properties)
    cls.__index__ = index
    return cls

# make mode subclasses
for m, i in MODE_INDEX.items():
    cls = _make_mode_class(m, i)
    globals()[cls.__name__] = cls
    __all__.append(cls.__name__)

def _cast_from(m):
    if isinstance(m, _mode):
        return m
    c = None
    subclasses = {c.__name__:c for c in _mode.__subclasses__()}
    if isinstance(m, str):
        c = subclasses.get(m, None)
    elif hasattr(m, "__iter__"):
        try:
            indx = sorted([MODE_PROPERTIES.index(i) for i in m])
        except:
            raise TypeError(f"Cannot cast {m} to a valid mode, given values must be in {MODE_PROPERTIES}")
        c_str = "".join(MODE_PROPERTIES_ALIAS[i] for i in indx)
        c = subclasses.get(c_str, None)
    if c is not None:
        return c
    raise TypeError(f"Cannot cast {m} to a valid mode.")



