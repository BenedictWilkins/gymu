#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
 Created on 18-02-2021 09:50:06

 [Description]
"""
__author__ ="Benedict Wilkins"
__email__ = "benrjw@gmail.com"
__status__ ="Development"

import itertools
import gym
import ray

from .. import mode as m
from ..policy import Uniform as uniform_policy

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
        elif not isinstance(env, gym.Env) and callable(env):
            env = env() # create a new environment # useful for parallelism (to avoid copy issues)

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

def episodes(env, policy=None, mode=m.s, n=1, max_length=1000, num_workers=1):
    assert num_workers >= 1
    if num_workers == 1:
        return Episodes(env, policy=policy, mode=mode, n=n, max_length=max_length)
    else:
        assert n >= num_workers
        workers = [EpisodesWorker.remote(env, policy=policy, mode=mode, n = n // num_workers, max_length=max_length) for _ in range(num_workers)]
        return ray.util.iter.from_actors(workers).gather_async()

class FuncRepeat:
    
    def __init__(self, fun, n=1):
        self.fun = fun
        self.n = n
        if self.n < 0:
            self.n = float("inf")
    
    def __iter__(self):
        print(self.n)
        for i in range(self.n):
            yield self.fun()
    
    def __len__(self):
        return self.n
        
class Episodes(FuncRepeat):

    def __init__(self, env, policy= None, mode=m.s, n=1, max_length=1000):
        assert isinstance(env, str) or callable(env) # avoid copy issues
                          
        def _episode():
            return episode(env, policy, mode=mode, max_length=max_length)
        super(Episodes, self).__init__(_episode, n=n) 

@ray.remote
class EpisodesWorker(ray.util.iter.ParallelIteratorWorker):
    
    def __init__(self, *args, **kwargs):
        super().__init__(Episodes(*args, **kwargs), False)
