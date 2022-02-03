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
from numpy.lib.arraysetops import isin
import ray
from tqdm.auto import tqdm

from .. import mode as m
from ..policy import Uniform as uniform_policy

class iterator:

    def __init__(self, env, policy=None, mode=m.s):
        if isinstance(env, str):
            env = gym.make(env)
        elif not isinstance(env, gym.Env) and callable(env):
            env = env() # create a new environment, useful for parallelism (to avoid copy issues)

        self.env = env
        if policy is None:
            policy = uniform_policy(self.env.action_space)
        self.policy = policy
        self.mode = m.mode(mode) # cast to correct type if not already a mode type
        
    def __iter__(self):
        state = self.env.reset()
        done = False
        while not done:
            action = self.policy(state)
            # state, action, next_state, reward, done, info
            result = [state, action, *self.env.step(action)] 
            yield self.mode(*[result[i] for i in self.mode.__index__])
            state = result[2]

            done = result[4]

def stream(env, policy=None, mode=m.s, max_episode_length=-1):
    """ Stream the environment, resetting whenever needed (or according to max_episode_length).

    Args:
        env (gym.Env): environment.
        policy (Policy, optional): policy. Defaults to a uniform random policy.
        mode (mode, optional): mode (see gym_mp.mode). Defaults to state mode.
        max_episode_length (int, optional): maximum length of the episode (longer episodes will be cut short). Defaults to -1 (infinite).
    """
    iter = iterator(env, policy, mode=mode)
    if max_episode_length > 0: # isslice doesnt reset automatically... (its a generator?)
        while True:
            for x in itertools.islice(iter, 0, max_episode_length):
                yield x
    else:
        while True:
            for x in iter:
                yield x


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

def episodes(env, policy=None, mode=m.s, n=1, max_length=1000, workers=1):
    assert workers >= 1
    if workers == 1:
        return Episodes(env, policy=policy, mode=mode, n=n, max_length=max_length)
    else:
        if not ray.is_initialized():
            ray.init()
        assert n >= workers
        assert isinstance(env, str) or callable(env) # env must be str or callable to support multiprocessing.
        workers = [EpisodesWorker.remote(env, policy=policy, mode=mode, n = int(n // workers), max_length=max_length) for _ in range(workers)]
        return tqdm(ray.util.iter.from_actors(workers).gather_async(), total=n)

class FuncRepeat:
    
    def __init__(self, fun, n=1):
        self.fun = fun
        self.n = n
    
    def __iter__(self):
        if self.n < 0:
            while True:
                yield self.fun()
        else:
            for i in range(self.n):
                yield self.fun()
    
    def __len__(self):
        return self.n

def make_environment(env):
    if isinstance(env, (gym.Env, gym.Wrapper)):
        return env
    elif isinstance(env, str):
        return gym.make(env)
    elif callable(env):
        return env()
    else:
        raise ValueError(f"Invalid enironment {env}.")
      
class Episodes(FuncRepeat):

    def __init__(self, env, policy=None, mode=m.s, n=1, max_length=1000):   
        env = make_environment(env)
        def _episode():
            return episode(env, policy, mode=mode, max_length=max_length)
        super(Episodes, self).__init__(_episode, n=n) 

@ray.remote
class EpisodesWorker(ray.util.iter.ParallelIteratorWorker):
    
    def __init__(self, *args, **kwargs):
        super().__init__(Episodes(*args, **kwargs), False)

@ray.remote
class StreamWorker(ray.util.iter.ParallelIteratorWorker):

    def __init__(self, env, policy, mode=m.s, max_episode_length=-1, **kwargs):
        env = make_environment(env)
        stream_it = stream(env, policy, mode=mode, max_episode_length=max_episode_length, **kwargs)
        super().__init__(stream_it, False)

# TODO ??? this in favour of ray?? 
def vectorize(envf, n=10, sync=False, shared_memory=False, daemon=True):
    if isinstance(envf, str):
        envf = lambda env=envf: gym.make(env)

    if sync:
        return gym.vector.SyncVectorEnv([envf] * n)
    else:
        return gym.vector.AsyncVectorEnv([envf] * n, shared_memory=shared_memory, daemon=daemon) 

