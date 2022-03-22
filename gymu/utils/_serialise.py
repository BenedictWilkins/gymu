#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" 
   Created on 17-03-2022
"""
__author__ = "Benedict Wilkins"
__email__ = "benrjw@gmail.com"
__status__ = "Development"

import numpy as np
import yaml
import gym
from gym.spaces import Discrete, Box

def _get_classname_tag(cls):
    module = cls.__module__
    name = cls.__qualname__
    if module is not None and module != "__builtin__":
        name = module + "." + name
    return f"!{name}"

def gym_spaces_discrete_representer(dumper, data):
    args = dict(n=data.n) # ignore start argument...
    return dumper.represent_mapping(_get_classname_tag(Discrete), args)
yaml.add_representer(Discrete, gym_spaces_discrete_representer)

def gym_spaces_discrete_constructor(loader, data):
    return gym.spaces.Discrete(**loader.construct_mapping(data))
yaml.SafeLoader.add_constructor(_get_classname_tag(Discrete), gym_spaces_discrete_constructor)

def gym_spaces_box_representer(dumper, data):
    low, high = data.low.ravel()[0], data.high.ravel()[0]
    assert np.all(data.low == low)   # non-constant low array not supported...
    assert np.all(data.high == high) # non-constant high array not supported...
    dtype = data.dtype.str
    return dumper.represent_mapping(_get_classname_tag(Box), dict(shape=list(data.shape), low=low.item(), high=high.item(), dtype=dtype))
yaml.add_representer(Box, gym_spaces_box_representer)

def gym_spaces_box_constructor(loader, data):
    data = loader.construct_mapping(data, deep=True)
    data['dtype'] = np.dtype(data['dtype'])
    return gym.spaces.Box(**data)
yaml.SafeLoader.add_constructor(_get_classname_tag(Box), gym_spaces_box_constructor)

def gym_env_representer(dumper, data):
    pass # TODO

def gym_env_constructor(loader, data):
    pass # TODO




if __name__ == "__main__":
    space = gym.spaces.Discrete(2)
    x = yaml.safe_load(yaml.dump(space))
    print(x)
    space = gym.spaces.Box(shape=(10,10), low=0, high=1)
    x = yaml.dump(space)
    print(x)
    y =  yaml.safe_load(x)
    print(y)

  






