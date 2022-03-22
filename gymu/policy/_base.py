#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
 Created on 14-01-2021 12:18:35

 [Description]
"""
__author__ ="Benedict Wilkins"
__email__ = "benrjw@gmail.com"
__status__ ="Development"

from abc import ABC, abstractmethod
import gym

__all__ = ("Policy", "DiscretePolicy")

class Policy(ABC):

    def __init__(self, action_space):
        self.action_space = action_space
    
    def __call__(self, *args, **kwargs):
        return self.sample(*args, **kwargs)

    def __str__(self):
        return "{0} - {1}".format(self.__class__.__name__, str(self.action_space))

    def __repr__(self):
        return str(self)

    @abstractmethod
    def sample(self, *args, **kwargs):
        pass

class DiscretePolicy(Policy):

    def __init__(self, action_space):
        if isinstance(action_space, int):
            action_space = gym.spaces.Discrete(action_space)
        assert isinstance(action_space, gym.spaces.Discrete)
        super(DiscretePolicy, self).__init__(action_space)
    
class ContinuousPolicy(Policy): # TODO

    def __init__(self, action_space):
        super(ContinuousPolicy, self).__init__(action_space)
