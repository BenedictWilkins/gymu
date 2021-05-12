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

from collections import deque

from ..spaces import NumpyBox


def _wrap(fun):
    def wrap(*args, **kwargs):
        w = fun(*args, **kwargs)
        w._wrap_type = fun.__name__[0].upper() + fun.__name__[1:]
        return w
    return wrap

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
        self._wrap_type = "Numpy"
        self._transform = lambda x : x # identity

    def __str__(self):
        return "<{0}{1}>".format(str(self._wrap_type), str(self.env))

    def __repr__(self):
        return str(self)

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

    @_wrap
    def transpose(self, axes):
        wrap = NumpyWrapper(self)
        wrap.observation_space = wrap.observation_space.transpose(axes)
        wrap._transform = lambda x: np.transpose(x, axes=axes)
        return wrap
     
    @_wrap
    def astype(self, dtype):
        wrap = NumpyWrapper(self)
        wrap.observation_space = wrap.observation_space.astype(dtype)
        wrap._transform = lambda x: x.astype(dtype)
        return wrap

    @_wrap
    def prod(self, axis=None, dtype=None, **kwargs):
        wrap = NumpyWrapper(self)
        wrap._transform = lambda x: np.prod(x, axis=axis, dtype=dtype, **kwargs)
        wrap.observation_space = wrap.observation_space.prod(axis=axis, dtype=dtype, **kwargs)
        return wrap

    @_wrap
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

class Stack(gym.Wrapper):

    """
        Frame stack wrapper.
    """

    def __init__(self, env, n=3, dim=0, lazy=False):
        super(Stack, self).__init__(env)
        ci = dim
        shape = list(self.observation_space.shape)

        shape[ci] = n * shape[ci]
        self.buffer = deque(maxlen=n)

        low = np.repeat(self.observation_space.low, n, axis=ci)
        high = np.repeat(self.observation_space.high, n, axis=ci)

        self.observation_space = gym.spaces.Box(low, high, dtype=self.observation_space.dtype)
        self.ci = ci

        assert not lazy # TODO

    def step(self, *args, **kwargs):
        state, *rest = self.env.step(*args, **kwargs)
        self.buffer.append(state)
        return (np.concatenate(self.buffer, axis=self.ci), *rest)

    def reset(self, *args, **kwargs):
        state = self.env.reset(*args, **kwargs)
        for i in range(self.buffer.maxlen):
            self.buffer.append(state)
        return np.concatenate(self.buffer, axis=self.ci)


class Temporal(NumpyWrapper): # TODO ... maybe?? 

    def __init__(self, env):
        super(Temporal, self).__init__(env)

    def __getitem__(self, i):
        if isinstance(i, int):
            raise IndexError("Invalid index: {0} must be a slice.".format(i))
        if isinstance(i, tuple) and isinstance(i[0], int):
            raise IndexError("Invalid index: {0} must be a slice.".format(i))

        if isinstance(i, slice):
            i = (i,)


        np_wrapper = super(Temporal, self).__getitem__(i[1:])


    def step(self, action):
        state, reward, done, *info = self.env.step(action)

    def reset(self):
        pass 
