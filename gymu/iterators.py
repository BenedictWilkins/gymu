#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
    Created on 16-09-2020 13:21:22

    [Description]
"""
__author__ = "Benedict Wilkins"
__email__ = "benrjw@gmail.com"
__status__ = "Development"

import itertools
import gym

from . import mode as m
from .policy import uniform as uniform_policy

def s_iterator(env, policy):
    state = env.reset()
    yield m.s(state)
    done = False
    while not done:
        action = policy(state)
        state, _, done, *_ = env.step(action)
        yield m.s(state)
        
def r_iterator(env, policy):
    state = env.reset()
    done = False
    while not done:
        action = policy(state)
        state, reward, done, *_ = env.step(action)
        yield m.r(reward)
        
def sa_iterator(env, policy):
    state = env.reset()
    done = False
    while not done:
        action = policy(state)
        nstate, _, done, *_ = env.step(action)
        yield m.sa(state, action)
        state = nstate
    yield m.sa(state, action)
        
def sr_iterator(env, policy):
    state = env.reset()
    done = False
    while not done:
        action = policy(state)
        nstate, reward, done, *_ = env.step(action)
        yield m.sr(state, reward)
        state = nstate
    #yield m.sr(state, 0.), True #?? maybe..
        
def ss_iterator(env, policy):
    state = env.reset()
    done = False
    while not done:
        action = policy(state)
        nstate, _, done, *_ = env.step(action)
        yield m.ss(state, nstate)
        state = nstate

def sar_iterator(env, policy):
    state = env.reset()
    done = False
    while not done:
        action = policy(state)
        nstate, reward, done, *_ = env.step(action)
        yield m.sar(state, action, reward)
        state = nstate
    
def ars_iterator(env, policy):
    state = env.reset()
    yield m.ars(None, None, state)
    done = False
    while not done:
        action = policy(state)
        state, reward, done, *_ = env.step(action)
        yield m.ars(action, reward, state)

def sas_iterator(env, policy):
    state = env.reset()
    done = False
    while not done:
        action = policy(state)
        nstate, _, done, *_ = env.step(action)
        yield m.sas(state, action, nstate)
        state = nstate

def sars_iterator(env, policy):
    state = env.reset()
    done = False
    while not done:
        action = policy(state)
        nstate, reward, done, *_ = env.step(action)
        yield m.sars(state, action, reward, nstate)
        state = nstate

class iterator:

    iterators = {m.s:s_iterator, m.r:r_iterator, m.sa:sa_iterator, m.ss:ss_iterator, m.sr:sr_iterator, 
             m.sar:sar_iterator, m.ars:ars_iterator, m.sas:sas_iterator, 
             m.sars:sars_iterator}

    def __init__(self, env, policy=None, mode=m.s):
        if isinstance(env, str):
            env = gym.make(env)
        self.env = env
        if policy is None:
            policy = uniform_policy(self.env.action_space)
        self.policy = policy
        self.mode = mode
        
    def __iter__(self):
        return iterator.iterators[self.mode](self.env, self.policy)

def episode(env, policy=None, mode=m.s, max_length=1000):
    """ 
        Creates an episode from the given environment and policy.

    Args:
        env (gym.Env): environment.
        policy (Policy, optional): policy. Defaults to a uniform random policy.
        mode (mode, optional): mode (see gym_mp.mode). Defaults to state mode.
        max_length (int, optional): maximum length of the episode (longer episodes will be cut short). Defaults to 1000.

    Returns:
        tuple([numpy.ndarray, ...]): episode as a collection of numpy ndarrays (1 for each mode component).
    """
    it = iterator(env, policy, mode=mode)
    it = itertools.islice(it, 0, max_length)
    return m.pack(it)

class episodes:

    def __init__(self, env, policy=None, mode=m.s, n=1, max_length=1000):
        self.env = env
        self.policy = policy
        self.mode = mode
        self.n = n
        if self.n < 0:
            self.n = float("inf")
        self.max_length = max_length

    def __iter__(self):
        return self

    def __next__(self):
        if self.n >= 0: 
            self.n -= 1
            return episode(self.env, self.policy, mode=self.mode, max_length=self.max_length)
        else:
            raise StopIteration()
        
    def __len__(self):
        return self.n