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

__all__ = ['pack', 'mode', #'Mode',
            "STATE", "ACTION", "REWARD", "NEXT_STATE", "DONE", "INFO", "OBSERVATION", "NEXT_OBSERVATION"]

# MODE_INDEX is based on the returned sequence of the gymu iterator step, which is, in order: state, action, reward, next_state, done, info
MODE_INDEX = dict(s=[0])
MODE_INDEX.update({k + "a":v + [1] for k,v in MODE_INDEX.items()})
MODE_INDEX.update({k + "r":v + [2] for k,v in MODE_INDEX.items()})
MODE_INDEX.update({k + "s":v + [3] for k,v in MODE_INDEX.items()})
MODE_INDEX.update({k + "d":v + [4] for k,v in MODE_INDEX.items()})
MODE_INDEX.update({k + "i":v + [5] for k,v in MODE_INDEX.items()}) 

STATE = "state"
ACTION = "action"
REWARD = "reward"
NEXT_STATE = "next_state"
DONE = "done"
INFO = "info"
OBSERVATION = "observation"
NEXT_OBSERVATION = "next_observation"

MODE_PROPERTIES = {STATE:0, ACTION:1, REWARD:2, NEXT_STATE:3, DONE:4, INFO:5, OBSERVATION:0, NEXT_OBSERVATION:3}
MODE_PROPERTIES_DEFAULT = (STATE, ACTION, REWARD, NEXT_STATE, DONE, INFO)
MODE_PROPERTIES_ALIAS = ('s', 'a', 'r', 's', 'd', 'i')

class _mode(tuple):

    def __new__(cls, *data, **kwdata):
        if len(data) > 0 and len(kwdata) > 0:
            raise ValueError("Mode arguments cannot be combined, specify either data or kwdata.")
        
        if len(kwdata) > 0: # construct from dictionary
            keys = tuple([MODE_PROPERTIES_DEFAULT[i] for i in cls.__index__])
            if len(keys) != len(kwdata.keys()):
                raise ValueError(f"Invalid keys {list(kwdata.keys())} specified for mode {cls}, must match {list(keys)}.")
            data = [x[1] for x in sorted(kwdata.items(), key = lambda x: MODE_PROPERTIES[x[0]])]

        if len(data) != len(cls.__index__):
            keys = tuple([MODE_PROPERTIES[i] for i in cls.__index__])
            if len(data) > 0 and isinstance(data[0], (list, tuple, dict)): 
                raise ValueError(f"Invalid argument type {type(data[0])} did you forget to unpack with * or ** ?")
            raise ValueError(f"Requires {len(cls.__index__)} arguments to match {keys}, only {len(data)} were supplied.")
        
        return tuple.__new__(cls, data)

    def __getitem__(self, i):
        if isinstance(i, int):
            return super().__getitem__(i)
        elif isinstance(i, str):
            return dict(self.items())[i] # TODO a bit slow...

    def __str__(self):
        return "mode-{0}".format(self.__class__.__name__)

    def __repr__(self):
        return str(self)

    def items(self):
        return zip(self.keys(), self.values())
    
    def values(self):
        return tuple([x for x in self])

    @classmethod
    def keys(cls):
        return tuple([MODE_PROPERTIES_DEFAULT[i] for i in cls.__index__])

    

def mode(m):
    return _cast_from(m)

def _cast_from(m):
    try: 
        if issubclass(m, _mode):
            return m
    except TypeError:
        pass # ignore, m is probably an instance
    c = None
    subclasses = {c.__name__:c for c in _mode.__subclasses__()}
    if isinstance(m, str):
        if not m in subclasses:
            raise TypeError(f"Cannot cast {m} to a valid mode, invalid mode type: {c_str}.")
        return subclasses.get(m)
    elif hasattr(m, "__iter__"):
        try:
            # assumed that m is an iterator of mode keys that match those in MODE_PROPERTIES
            indx = sorted([MODE_PROPERTIES[x] for x in m]) # important that they are sorted...     
        except:
            raise TypeError(f"Cannot cast {m} to a valid mode, given values must be in {MODE_PROPERTIES}")
        c_str = "".join(MODE_PROPERTIES_ALIAS[i] for i in indx)
        if not c_str in subclasses:
            raise TypeError(f"Cannot cast {m} to a valid mode, invalid mode type: {c_str}.")
        return subclasses.get(c_str)
    else: 
        raise TypeError(f"Cannot cast {m} to a valid mode.")

#Mode = _mode # type alias 

def pack(modes):
    """ Pack a list of modes into numpy arrays. """
    modes = list(modes)
    mode = type(modes[0])
    return mode(*[np.array(d) for d in [i for i in zip(*modes)]])

# MAKE MODE CLASSES
def _make_mode_class(name, index):
    prop_map = dict(s='state', a='action', r='reward', d='done', i='info')
    properties = {}
    for i, n in enumerate(name):
        properties[prop_map[n]] = property(lambda self, j=i: self[j])
        if n == 's': 
            prop_map['s'] = 'next_state'
    cls = type(m, (_mode,), properties)
    cls.__index__ = index
    #print(cls, cls.__index__, cls.keys())
    return cls

for m, i in MODE_INDEX.items():
    cls = _make_mode_class(m, i)
    globals()[cls.__name__] = cls
    __all__.append(cls.__name__)

    

            








if __name__ == "__main__":
    subclasses = {c.__name__:c for c in _mode.__subclasses__()}
    print("-- BUILD FROM LIST --")
    for m,i in MODE_INDEX.items():
        print(list(subclasses[m](*i).items()))

    print("-- BUILD FROM DICT --")
    for m,i in MODE_INDEX.items():
        d = {MODE_PROPERTIES_DEFAULT[j]:MODE_PROPERTIES[MODE_PROPERTIES_DEFAULT[j]] for j in i}
        print(list(subclasses[m](**d).items()))

    print(list(s(observation=1).items()))
