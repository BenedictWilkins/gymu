#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
 Created on 14-01-2021 12:11:59

 [Description]
"""
__author__ ="Benedict Wilkins"
__email__ = "benrjw@gmail.com"
__status__ ="Development"

from abc import ABC, abstractmethod
import numpy as np
import gym

from .. import spaces
from .base import *

__all__ = []

try:
    import torch as _pytorch
    from . import torch
except:
    pass 

class Uniform(DiscretePolicy):

    def __init__(self, action_space):
        super(Uniform, self).__init__(action_space)

    def sample(self, *args, **kwargs):
        return self.action_space.sample()

'''
def onehot(policy, dtype=np.float32):
    assert np.issubdtype(policy.action_space.dtype, np.integer)
    assert isinstance(policy.action_space, gym.spaces.Discrete)

    sample =  policy.sample
    policy.action_space = spaces.OneHot(policy.action_space.n, dtype)

    def oh(x):
        r = np.zeros(policy.action_space.shape[0], dtype=policy.action_space.dtype)
        r[sample(x)] = 1
        return r
    policy.sample = oh
    
    return policy

  

def random(action_space, p=None, dtype=np.int64): #TODO assume discrete action_space?
    """ Random policy that selects an action from the given (discrete) action space according to the given probabilities p.

    Args:
        action_space (gym.spaces.Discrete, int): action space
        p (sequence, optional): action probabilities associated with each action. Defaults to uniform probability.
        dtype (type, optional): dtype of a sampled action. Defaults to np.int64.

    Returns:
        DiscretePolicy: the policy
    """
    if p is None:
        return uniform(action_space, dtype=dtype)
    
    policy = DiscretePolicy(action_space, dtype=dtype)
    assert len(p) == policy.action_space.n
    sample_space = np.arange(0, policy.action_space.n)
    policy.sample = lambda *args, **kwargs: action_space.dtype(np.random.choice(sample_space, p=p))
    return policy
    

# TODO update others to follow DiscretePolicy!

def e_greedy_policy(action_space, critic, epsilon=0.01, onehot=False): 
    def __policy(state):
        if np.random.uniform() > epsilon:
            return action_space.sample()
        else:
            return np.argmax(critic(state))
    
    policy = __policy 
    
    if onehot:
        policy = onehot_policy(policy, action_space.n)
    return policy

def probabilistic_policy(action_space, actor, onehot=False):
    actions = np.arange(action_space.n)
    def __policy(s):
        return np.random.choice(actions, p = actor(s))
    policy = __policy
    if onehot:
        policy = onehot_policy(policy, action_space.n)
    return policy

if __name__ == "__main__":
    from gym.spaces.discrete import Discrete
    action_space = Discrete(3)
    
    
    policy = e_greedy_policy(action_space)
    
'''
    
    
    
    