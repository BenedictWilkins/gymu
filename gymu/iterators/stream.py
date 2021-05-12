#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
    Created on 17-03-2021 13:56:01
    Gym iterators that ignore the done flag, can be used for non-episodic environments, 
    as well as gym.vector environments.
"""
__author__ ="Benedict Wilkins"
__email__ = "benrjw@gmail.com"
__status__ ="Development"


import itertools
import gym
from .. import mode as m
from ..policy import Uniform as uniform_policy

def s_iterator(env, policy):
    state = env.reset()
    yield m.s(state)
    while True:
        action = policy(state)
        state, _, _, *_ = env.step(action)
        yield m.s(state)
        
def r_iterator(env, policy):
    state = env.reset()
    while True:
        action = policy(state)
        state, reward, _, *_ = env.step(action)
        yield m.r(reward)
        
def sa_iterator(env, policy):
    state = env.reset()
    while True:
        action = policy(state)
        nstate, _, _, *_ = env.step(action)
        yield m.sa(state, action)
        state = nstate
    yield m.sa(state, action)
        
def sr_iterator(env, policy):
    state = env.reset()
    while True:
        action = policy(state)
        nstate, reward, _, *_ = env.step(action)
        yield m.sr(state, reward)
        state = nstate
    #yield m.sr(state, 0.), True #?? maybe..
        
def ss_iterator(env, policy):
    state = env.reset()
    while True:
        action = policy(state)
        nstate, _, _, *_ = env.step(action)
        yield m.ss(state, nstate)
        state = nstate

def sar_iterator(env, policy):
    state = env.reset()
    while True:
        action = policy(state)
        nstate, reward, _, *_ = env.step(action)
        yield m.sar(state, action, reward)
        state = nstate
    
def ars_iterator(env, policy):
    state = env.reset()
    yield m.ars(None, None, state)
    while True:
        action = policy(state)
        state, reward, _, *_ = env.step(action)
        yield m.ars(action, reward, state)

def sas_iterator(env, policy):
    state = env.reset()
    while True:
        action = policy(state)
        nstate, _, _, *_ = env.step(action)
        yield m.sas(state, action, nstate)
        state = nstate

def sars_iterator(env, policy):
    state = env.reset()
    while True:
        action = policy(state)
        nstate, reward, _, *_ = env.step(action)
        yield m.sars(state, action, reward, nstate)
        state = nstate

class iterator:

    iterators = {m.s:s_iterator, m.r:r_iterator, m.sa:sa_iterator, m.ss:ss_iterator, m.sr:sr_iterator, 
             m.sar:sar_iterator, m.ars:ars_iterator, m.sas:sas_iterator, 
             m.sars:sars_iterator}

    def __init__(self, env, policy=None, mode=m.s):
        if isinstance(env, str):
            env = gym.make(env)
        elif not isinstance(env, gym.Env) and callable(env):
            env = env()

        self.env = env
        if policy is None:
            policy = uniform_policy(self.env.action_space)
        self.policy = policy
        self.mode = mode
        
    def __iter__(self):
        return iterator.iterators[self.mode](self.env, self.policy)