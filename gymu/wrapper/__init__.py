#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
    Created on 18-09-2020 11:03:23

    Wrappers from gym environments.
"""
__author__ = "Benedict Wilkins"
__email__ = "benrjw@gmail.com"
__status__ = "Development"

import copy
import gym
import numpy as np

from collections import deque
from types import SimpleNamespace
import itertools

try:
    import skimage.transform
except:
    pass 

from ..spaces import NumpyBox
from .. import iterators
from .. import mode

from . import image
from .np_wrap import NumpyWrapper
from .image import image 

# TODO move to NumpyWrapper? 
class Stack(gym.Wrapper):

    """
        Frame stack wrapper.
    """

    def __init__(self, env, n=3):
        super(Stack, self).__init__(env)
        shape, fshape = __image_format__(env.observation_space)
        ci = fshape.index("C") # default HWC
        shape = list(shape)

        shape[ci] = n * shape[ci]
        self.__buffer = deque(maxlen=n)
        self.observation_space = gym.spaces.Box(self.observation_space.low.flat[0], self.observation_space.high.flat[0], shape=shape, dtype=env.observation_space.dtype)
        self.__ci = ci


    def step(self, *args, **kwargs):
        state, *rest = self.env.step(*args, **kwargs)
        self.__buffer.append(state)
        return (np.concatenate(self.__buffer, axis=self.__ci), *rest)

    def reset(self, *args, **kwargs):
        state = self.env.reset(*args, **kwargs)
        for i in range(self.__buffer.maxlen):
            self.__buffer.append(state)
        return np.concatenate(self.__buffer, axis=self.__ci)

class ActionDelay(gym.ActionWrapper):

    """
        Delays action by n steps.
    """

    def __init__(self, env, initial_action=4, delay=0):
        super(ActionDelay, self).__init__(env)
        if isinstance(initial_action, (int, float)):
            initial_action = [initial_action] * (delay + 1)
        elif isinstance(initial_action, np.ndarray):
            initial_action = initial_action.tolist() + [0]
        assert len(initial_action) - 1 == delay
        
        self.action_buffer = deque(initial_action, maxlen=delay+1)
        self.delay = delay
        
    def action(self, a):
        self.action_buffer.append(a)
        return self.action_buffer[0]


        





