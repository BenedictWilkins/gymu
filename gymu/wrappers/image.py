#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" 
   Created on 22-03-2022
"""
__author__ = "Benedict Wilkins"
__email__ = "benrjw@gmail.com"
__status__ = "Development"

import gym
import numpy as np

__all__ = ("Float", "Integer", "HWC", "CHW")

class Float(gym.ObservationWrapper):

    def __init__(self, env):
        super().__init__(env)
        assert isinstance(env.observation_space, gym.spaces.Box)
        assert issubclass(env.observation_space.dtype.type, np.integer)
        self._low = env.observation_space.low.ravel()[0]
        self._high = env.observation_space.high.ravel()[0]
        assert np.all(env.observation_space.low.ravel() == self._low)
        assert np.all(env.observation_space.high.ravel() == self._high)
        self.observation_space = gym.spaces.Box(low=0., high=1., dtype=np.float32, shape=env.observation_space.shape)
        
    def observation(self, x):
        x = x.astype(np.float32)
        x -= self._low
        x /= (self._high - self._low)
        return x

class Integer(gym.ObservationWrapper):

    def __init__(self, env):
        super().__init__(env)
        assert isinstance(env.observation_space, gym.spaces.Box)
        assert issubclass(env.observation_space.dtype.type, np.floating)
        self._low = env.observation_space.low.ravel()[0]
        self._high = env.observation_space.high.ravel()[0]
        assert np.all(env.observation_space.low.ravel() == self._low)
        assert np.all(env.observation_space.high.ravel() == self._high)
        self.observation_space = gym.spaces.Box(low=0, high=255, dtype=np.uint8, shape=env.observation_space.shape)
        
    def observation(self, x):
        x -= self._low
        x /= (self._high - self._low)
        x *= 255.
        return x.astype(np.uint8)

class CHW(gym.ObservationWrapper):

    def __init__(self, env):
        super().__init__(env)
        assert len(env.observation_space.shape) == 3
        _shape = self.observation(env.observation_space.low).shape
        _low = env.observation_space.low.ravel()[0]
        _high = env.observation_space.high.ravel()[0]
        _dtype = env.observation_space.dtype
        assert np.all(env.observation_space.low.ravel() == _low)
        assert np.all(env.observation_space.high.ravel() == _high)
        self.observation_space = gym.spaces.Box(low=_low, high=_high, shape=_shape, dtype=_dtype)
        
    def observation(self, x):
        return x.transpose((2,0,1))

class HWC(gym.ObservationWrapper):

    def __init__(self, env):
        super().__init__(env)
        assert len(env.observation_space.shape) == 3
        _shape = self.observation(env.observation_space.low).shape
        _low = env.observation_space.low.ravel()[0]
        _high = env.observation_space.high.ravel()[0]
        _dtype = env.observation_space.dtype
        assert np.all(env.observation_space.low.ravel() == _low)
        assert np.all(env.observation_space.high.ravel() == _high)
        self.observation_space = gym.spaces.Box(low=_low, high=_high, shape=_shape, dtype=_dtype)
        
    def observation(self, x):
        return x.transpose((1,2,0))





