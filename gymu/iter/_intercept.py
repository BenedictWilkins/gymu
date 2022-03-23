#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" 
   Created on 21-03-2022

   Sometimes it is useful to get data from an environment that has been wrapped. This module makes this easier, simply wrap the environment at the point of interest with an InterceptWrapper and use an InterceptIterator to get the data. Data will appear in the final information as info = {"itercepted" = [<INTERCEPTED_DATA>]}.

   WARNING: This does not currently work with multi process/threaded wrappers.
"""
__author__ = "Benedict Wilkins"
__email__ = "benrjw@gmail.com"
__status__ = "Development"

from typing import Iterable, Union
import gym

from . import Iterator

__all__ = ("InterceptWrapper", "InterceptIterator", "interceptable")

def InterceptWrapper(env, unwrapped=False):
    """ Intercept data from a wrapped environment.

    Args:
        env (Union[gym.Wrapper, gym.Env]): environment to intercept data from.
        unwrapped (bool, optional): whether to completely unwrap the environment, interscepting from the base environment. WARNING: this inserts itself in the wrapper hierarchy, it might break some parent wrappers. It is better to wrap the enviroment during the creation of the wrapper hierarchy. Defaults to False.
    """
    if unwrapped:
        envs = _find_wrappers(env, type(env.unwrapped))
        env = envs[-1]
        instance = _InterceptWrapper(env)
        if len(envs) > 1:
            wrapper = envs[-2]
            if hasattr(wrapper, 'env'):
                wrapper.env = instance
            return wrapper
    else:
        instance = _InterceptWrapper(env)
    return instance

class _InterceptWrapper(gym.Wrapper):
    
    def __init__(self, env : Union[gym.Wrapper, gym.Env]):
        """ Intercept data from a wrapped environment.

        Args:
            env (Union[gym.Wrapper, gym.Env]): environment to intercept data from.
        """
        super().__init__(env)
        self.data = []
        
    def step(self, action):
        state, reward, done, info = self.env.step(action)
        self.data.append((state, action, reward, done, info))
        #print("INTERCEPT:", state, action, done)
        return state, reward, done, info
        
    def reset(self):
        x = self.env.reset()
        self.data.append((x,None,None,None,None))
        return x
    
    def intercept(self): # called by the InterceptIterator
        data, self.data = self.data, []
        return data

def interceptable(env):
    """ Checks if an environment is wrapped with the InterceptWrapper at some point in its wrapper hierarchy.

    Args:
        env (Union[gym.Env, gym.Wrapper]): gym environment to test.

    Returns:
        bool: True if an InterceptWrapper was found, False otherwise.
    """
    try:
        _find_wrappers(env, _InterceptWrapper)
        return True
    except:
        return False

class InterceptIterator(Iterable):

    def __init__(self, env, *args, intercept_only=True, **kwargs):
        super().__init__()
        self._iterator = Iterator(env, *args, **kwargs)

        self._intercept_env = _find_wrappers(env, _InterceptWrapper)[-1]
        self._intercept_only = intercept_only
        self.state = None
       
    @property
    def mode(self):
        return self._iterator.mode
    
    @property
    def env(self):
        return self._intercept_env 
    
    @property
    def env_wrapped(self):
        return self._iterator.env

    @property
    def policy(self):
        return self._iterator.policy

    @property
    def max_length(self):
        return self._iterator.max_length

    def __iter__(self):
        if self._intercept_only:
            return self._intercept_only_iter()
        else:
            return self._iter()
            
    def _intercept_only_iter(self):
        _iterator = iter(self._iterator)
        x = next(_iterator)
        yield from self._intercept_initial(self._intercept_env.intercept())
        for x in _iterator:
            yield from self._intercept(self._intercept_env.intercept())

    def _iter(self):
        _iterator = iter(self._iterator)
        x = next(_iterator)
        yield x, [y for y in self._intercept_initial(self._intercept_env.intercept())]
        for x in _iterator:
            yield x, [y for y in self._intercept(self._intercept_env.intercept())]
    
    def _intercept_initial(self, interception):
        self.state, *_ = interception[0]
        yield from self._intercept(interception[1:])

    def _intercept(self, interception):
        for next_state, action, reward, done, info in interception:
            result = (self.state, action, reward, next_state, done, info) # S_t, A_t, R_{t+1}, S_{t+1}, done_{t+1}, info_{t+1}
            #print("ITER", result)
            yield self.mode(*[result[i] for i in self.mode.__index__])
            self.state = next_state
            if done:
                break # sometimes a reset is called during step by a wrapper, this can mess things up a bit... dont do this in your wrappers!

def _find_wrappers(env, cls): # find all wrappers up until type 'cls'
    envs = [env]
    while not isinstance(env, cls):
        #print(env, cls)
        if hasattr(env, "env"):
            env = env.env # unwrap
        elif hasattr(env, "envs"):  # TODO doesnt properly support vec environments... this is from stable_baselines3... i wish they followed the gym api...
            assert len(env.envs) == 1 
            env = env.envs[0] # unwrap
        elif hasattr(env, "venv"):  # TODO doesnt properly support vec environments... this is from stable_baselines3... i wish they followed the gym api...
            env = env.venv # unwrap
        else:
            raise ValueError(f"Failed to unwrap '{env}' in wrapper hierarchy.")
        envs.append(env)
    return envs