#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
    Created on 16-09-2020 13:13:31

    [Description]
"""
__author__ = "Benedict Wilkins"
__email__ = "benrjw@gmail.com"
__status__ = "Development"

import copy
import gym
import ray
import itertools

from . import iterators
from . import mode
from . import policy
from . import spaces
from . import wrappers

__all__ = ('iterators', 'mode', 'policy', 'spaces', 'wrappers')

from .iterators import episode, episodes, iterator


def stack(states, frames=3, copy=True):
    """ Stack states. with an input of states [s1, s2, ..., sn] where each state is a grayscale image, 
        the output will is given as:
        
        [[s1, s2, ..., sm  ],
         [s2, s3, ..., sm+1],
                  ...  sn  ]]
        where m = frames


        Args:
            states (numpy.ndarray): states [s1, s2, ..., sn] as images NHWC format.
            frames (int, optional): number of states to stack. Defaults to 3.
            copy (bool, optional): create a new array to store the ouput. WARNING: without a copy, modifying 
                                    the array can have undesirable effects as stride_tricks is used. Defaults to True.

        Returns:
            numpy.ndarray: stacked states
    """
    states = states.squeeze() #remove channels
    shape = states.shape
    assert len(shape) == 3 # NHW format

    # stack frames
    stacked = np.lib.stride_tricks.as_strided(states, shape=(shape[0] - frames, *shape[1:], frames), strides=(*states.strides, states.strides[0]))
    if copy:
        stacked = np.copy(stacked)
    return stacked



def init():
    ray.init()

def mp_episodes(env, policy=lambda: None, mode=mode.s, workers=1, n=1, max_length=1000):
    """ 
        An iterator for episodes, generates n episodes in parallel.

        env (gym.Env): environment.
        policy (Policy, optional): policy. Defaults to a uniform random policy.
        mode (mode, optional): mode (see gym_mp.mode). Defaults to state mode.
        workers (int, optional): number of environments used to generate the episodes. Defaults to 1.
        n (int, optional): number of episodes to generate (per worker), a negative value corresponds to infinite episodes. Defaults to 1.
        max_length (int, optional): maximum length of an episode (longer episodes will be cut short). Defaults to 1000.
    """
    if isinstance(env, str):
        env = lambda x=env: gym.make(x)

    class iterator: 
        def __iter__(self):
            return iterators.episodes(env(), policy(), mode=mode, n=n, max_length=max_length)

    it = ray.util.iter.from_iterators([iterator() for _ in range(workers)])
    return it

def async_iterator(env, policy=lambda: None, mode=mode.s, workers=2, repeat=False):
    """ 
        Creates an environment iterator that will iterate over many environments in parallel. 
        The result from each env.step will be returned aysnchronously (without order) and works on a 
        first ready basis.

        This is useful for collecting batches comprised of states/actions/rewards from independant environments.

        Example:

            env = gym.make("CartPole-v0")
            action_space = env.action_space

            policy_f = lambda : policy.uniform(action_space)
            env_f = lambda : gym.make("CartPole-v0")

            it = async_iterator(env_f, policy_f, workers=2)
            for state in it:
                print(state)

            >> env1.state0
            >> env1.state1
            >> env2.state0
            >> env1.state2
            >> env2.state1
            >> ...

    Args:
        env (fun, str): factory for a Gym environment (or environment name)
        policy (fun, optional): factory for a policy. Defaults to a uniform random policy.
        mode (mode, optional): mode (see gym_mp.mode). Defaults to mode.s.
        workers (int, optional): number of environments. Defaults to 2.
        repeat (bool, optional): repeat the iterators forever. Defaults to False.

    Returns:
        iter: an async iterator
    """
    return mp_iterator(env, policy, mode=mode, workers=workers, repeat=repeat).gather_async()


def sync_iterator(env, policy, mode=mode.s, workers=2):
    """ 
        Creates an environment iterator that will iterate over many environments in parallel. 
        The result from each env.step will be returned ysnchronously (in order).

        This is useful for collecting batches comprised of states/actions/rewards from independant environments.

        Example:

            env = gym.make("CartPole-v0")
            action_space = env.action_space

            policy_f = lambda : policy.uniform(action_space)
            env_f = lambda : gym.make("CartPole-v0")

            it = sync_iterator(env_f, policy_f, workers=2)
            for state in it:
                print(state)

            >> env1.state0
            >> env2.state0
            >> env1.state1
            >> env2.state1
            >> env1.state2
            >> env2.state2
            >> ...

    Args:
        env (fun, str): factory for a Gym environment (or environment name)
        policy (fun, optional): factory for a policy. Defaults to a uniform random policy.
        mode (mode, optional): mode (see gym_mp.mode). Defaults to mode.s.
        workers (int, optional): number of environments. Defaults to 2.
        repeat (bool, optional): repeat the iterators forever. Defaults to False.

    Returns:
        iter: a local iterator
    """
    return mp_iterator(env, policy, mode=mode, workers=workers, repeat=repeat).gather_sync()

def mp_iterator(env, policy, mode=mode.s, workers=2, repeat=False):
    if isinstance(env, str):
        env = lambda : gym.make(env)

    class iterator: 
        def __iter__(self):
            return iter(iterators.iterator(env(), policy(), mode=mode))

    it = ray.util.iter.from_iterators([iterator() for _ in range(workers)], repeat=repeat)
    return it





