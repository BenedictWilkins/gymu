#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" 
   Created on 23-03-2022
"""
__author__ = "Benedict Wilkins"
__email__ = "benrjw@gmail.com"
__status__ = "Development"

import gym

class InfoResetWrapper(gym.Wrapper):
    """ 
        In older versions of gym the `env.reset` function does not return info. This wrapper will return an empty info dict on reset. 
        The issue can be found: https://github.com/openai/gym/issues/1683
        
        This makes an environment work properly with the gym iterators that expect `state, info = env.reset()`.
    """
    def step(self, a):
        return self.env.step(a)

    def reset(self):
        return self.env.reset(), dict() # return (empty) info as well!