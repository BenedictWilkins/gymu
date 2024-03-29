#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" 
   Created on 22-03-2022
"""
__author__ = "Benedict Wilkins"
__email__ = "benrjw@gmail.com"
__status__ = "Development"

import numpy as np
import gym

from collections import deque

__all__ = ('ActionDelay',)

class ActionDelay(gym.ActionWrapper):
    """
        Wrapper that delays action by n steps.
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
