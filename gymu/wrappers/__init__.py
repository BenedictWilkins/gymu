#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
    Created on 18-09-2020 11:03:23

    Wrappers from gym environments.
"""
__author__ = "Benedict Wilkins"
__email__ = "benrjw@gmail.com"
__status__ = "Development"

from .np_wrap import *
from .image import * 

from . import atari

__all__ = ('atari',)

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
        





