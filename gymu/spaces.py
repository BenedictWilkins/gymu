#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
    Created on 16-09-2020 13:28:00

    [Description]
"""
__author__ = "Benedict Wilkins"
__email__ = "benrjw@gmail.com"
__status__ = "Development"

import gym
import numpy as np

class OneHot(gym.Space):

    def __init__(self, size, dtype=np.float32):
        assert isinstance(size, int) and size > 0
        self.size = size
        super(OneHot, self).__init__((size,), dtype)

    def sample(self):
        r = np.zeros(self.size, dtype=self.dtype)
        r[np.random.randint(self.size)] = 1
        return r

    def contains(self, x):
        if isinstance(x, (list, tuple, np.ndarray)) and len(x) == self.size:
            u, c = np.unique(x, return_counts=True)
            return u[0] == 0 and u[1] == 1 and c[int(u[0])] == self.size - 1
        else:
            return False

    def __repr__(self):
        return "OneHot(%d)" % self.size

class NumpyBox(gym.spaces.Box):
    
    def __init__(self, low, high, shape=None, dtype=np.float32):
        super(NumpyBox, self).__init__(low, high, shape=shape, dtype=dtype)
    
    def __getitem__(self, i):
        return NumpyBox(self.low[i], self.high[i], dtype=self.dtype)
    
    def astype(self, dtype):
        return NumpyBox(self.low.astype(dtype), self.high.astype(dtype), dtype=dtype)
       
    def transform(self, fun):
        low = fun(self.low)
        high = fun(self.high)
        assert low.dtype == high.dtype
        assert low.shape == high.shape
        return NumpyBox(low, high, dtype=low.dtype)
    
    def transpose(self, axes):
        return self.transform(lambda x: np.transpose(x, axes=axes))

    def prod(self, axis=None, dtype=None, **kwargs):
        return self.transform(lambda x: np.prod(x, axis=axis, dtype=dtype, **kwargs))
    
    def sum(self, axis=None, dtype=None, **kwargs):
        return self.transform(lambda x: np.sum(x, axis=axis, dtype=dtype, **kwargs))
    
    ## boolean operators should do something different, its now a bool box array? (__eq__ will not work properly here)

    def __gt__(self, x):
        return self.transform(lambda y : y > x)
    
    def __ge__(self, x):
        return self.transform(lambda y : y >= x)
    
    def __lt__(self, x):
        return self.transform(lambda y : y < x)
    
    def __le__(self, x):
        return self.transform(lambda y : y <= x)
    
    def __eq__(self, x):
        return self.transform(lambda y : y == x)
    
    def __ne__(self, x):
        return self.transform(lambda y : y != x)
     
    def __add__(self, x):
        return self.transform(lambda y : y + x)
    
    def __sub__(self, x):
        return self.transform(lambda y : y - x)
    
    def __mul__(self, x):
        return self.transform(lambda y : y * x)
        
    def __div__(self, x):
        return self.transform(lambda y : y / x)
    
    def __floordiv__(self, x):
        return self.transform(lambda y : y // x)
    
    def __truediv__(self, x):
        return self.transform(lambda y : y / x)
    
    def __neg__(self):
        return self.transform(lambda y : -y)
    
    def __abs__(self):
        return self.transform(lambda y : np.abs(y))