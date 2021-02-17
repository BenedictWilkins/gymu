#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
 Created on 17-02-2021 16:12:18

 [Description]
"""
__author__ ="Benedict Wilkins"
__email__ = "benrjw@gmail.com"
__status__ ="Development"

import numpy as np
import gym
from ..spaces import NumpyBox

class NumpyWrapper(gym.Wrapper):
    
    """
        Follows the numpy array API.
    """
    
    def __init__(self, env):
        super(NumpyWrapper, self).__init__(env)
        if isinstance(env.observation_space, NumpyBox):
            self.observation_space = env.observation_space
        elif isinstance(env.observation_space, gym.spaces.Box):
            self.observation_space = NumpyBox(env.observation_space.low, env.observation_space.high, dtype=env.observation_space.dtype)
        else:
            raise ValueError("Invalid space {0} must be of type {1}".format(type(env.observation_space), gym.spaces.Box))
        self._transform = lambda x: x # no transform

    def step(self, action, *args, **kwargs):
        observation, *rest = self.env.step(action, *args, **kwargs)
        observation = self._transform(observation)
        return (observation, *rest)

    def reset(self, *args, **kwargs):
        observation = self.env.reset(*args, **kwargs)
        observation = self._transform(observation)
        return observation
    
    def __getitem__(self, i):
        wrap = NumpyWrapper(self)
        wrap.observation_space = wrap.observation_space[i]
        wrap._transform = lambda x: x[i]
        return wrap

    def transpose(self, axes):
        wrap = NumpyWrapper(self)
        wrap.observation_space = wrap.observation_space.transpose(axes)
        wrap._transform = lambda x: np.transpose(x, axes=axes)
        return wrap
     
    def astype(self, dtype):
        wrap = NumpyWrapper(self)
        wrap.observation_space = wrap.observation_space.astype(dtype)
        wrap._transform = lambda x: x.astype(dtype)
        return wrap
    
    def prod(self, axis=None, dtype=None, **kwargs):
        wrap = NumpyWrapper(self)
        wrap._transform = lambda x: np.prod(x, axis=axis, dtype=dtype, **kwargs)
        wrap.observation_space = wrap.observation_space.prod(axis=axis, dtype=dtype, **kwargs)
        return wrap
    
    def sum(self, axis=None, dtype=None, **kwargs):
        wrap = NumpyWrapper(self)
        wrap._transform = lambda x: np.sum(x, axis=axis, dtype=dtype, **kwargs)
        wrap.observation_space = wrap.observation_space.sum(axis=axis, dtype=dtype, **kwargs)
        return wrap
    
    def __gt__(self, x):
        wrap = NumpyWrapper(self)
        wrap._transform = lambda y : y > x
        wrap.observation_space = wrap.observation_space > x 
        return wrap
    
    def __ge__(self, x):
        wrap = NumpyWrapper(self)
        wrap._transform = lambda y : y >= x
        wrap.observation_space = wrap.observation_space >= x 
        return wrap
    
    def __lt__(self, x):
        wrap = NumpyWrapper(self)
        wrap._transform = lambda y : y < x
        wrap.observation_space = wrap.observation_space < x
        return wrap
     
    def __le__(self, x):
        wrap = NumpyWrapper(self)
        wrap._transform = lambda y : y <= x
        wrap.observation_space = wrap.observation_space <= x 
        return wrap
    
    def __eq__(self, x):
        wrap = NumpyWrapper(self)
        wrap._transform = lambda y : y == x
        wrap.observation_space = wrap.observation_space == x
        return wrap
    
    def __ne__(self, x):
        wrap = NumpyWrapper(self)
        wrap._transform = lambda y : y != x
        wrap.observation_space = wrap.observation_space != x 
        return wrap
     
    def __add__(self, x):
        wrap = NumpyWrapper(self)
        wrap._transform = lambda y : y + x
        wrap.observation_space = wrap.observation_space + x 
        return wrap
    
    def __sub__(self, x):
        wrap = NumpyWrapper(self)
        wrap._transform = lambda y : y - x
        wrap.observation_space = wrap.observation_space - x
        return wrap
    
    def __mul__(self, x):
        wrap = NumpyWrapper(self)
        wrap._transform = lambda y : y * x
        wrap.observation_space = wrap.observation_space * x
        return wrap
        
    def __div__(self, x):
        wrap = NumpyWrapper(self)
        wrap._transform = lambda y : y / x
        wrap.observation_space = wrap.observation_space / x 
        return wrap
    
    def __floordiv__(self, x):
        wrap = NumpyWrapper(self)
        wrap._transform = lambda y : y // x
        wrap.observation_space = wrap.observation_space // x 
        return wrap 
    
    def __truediv__(self, x):
        wrap = NumpyWrapper(self)
        wrap._transform = lambda y : y / x
        wrap.observation_space = wrap.observation_space / x 
        return wrap 
    
    def __neg__(self):
        wrap = NumpyWrapper(self)
        wrap._transform = lambda y : -y
        wrap.observation_space = -wrap.observation_space
        return wrap 
    
    def __abs__(self):
        wrap = NumpyWrapper(self)
        wrap._transform = lambda y : np.abs(y)
        wrap.observation_space = abs(wrap.observation_space)
        return wrap 