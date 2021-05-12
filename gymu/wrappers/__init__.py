#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
    Created on 18-09-2020 11:03:23

    Wrappers from gym environments.
"""
__author__ = "Benedict Wilkins"
__email__ = "benrjw@gmail.com"
__status__ = "Development"

from .np_wrap import *
from .image import * 

from . import atari

__all__ = ('atari',)

def numpy(env):
    return NumpyWrapper(env)

class ActionDelay(gym.ActionWrapper):

    """
        Delays action by n steps.
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

class EpisodeRecordWrapper(gym.Wrapper): #TOPDO doesnt work currently (missing dependency -- just use h5py)

    def __init__(self, env, path, compress=True):
        assert False # NOT IMPLEMENTED PROPERLY YET
        
        super(EpisodeRecordWrapper, self).__init__(env)
        self.states = []
        self.actions = []
        self.rewards = []

        self.state_t = None
        self.reward_t = None

        self.path = path

        self.already_done = False

    def step(self, action_t):
        assert not self.already_done #dont save multiple times just because someone isnt calling reset!
         
        state, reward, done, info = self.env.step(action_t)
        state = np.copy(state)


        self.states.append(self.state_t)
        self.actions.append(action_t)
        self.rewards.append(self.reward_t)

        self.state_t = state
        self.reward_t = reward

        if done:
            self.states.append(self.state_t)
            self.actions.append(np.nan)
            self.rewards.append(self.reward_t)
            
            print("SAVING: {0:<5} frames (states, actions, rewards)".format(len(self.states)))
            fu_save(self.path, {"state":self.states,"action":self.actions,"reward":self.rewards}, overwrite=False, force=True)

            self.states.clear()
            self.actions.clear()
            self.rewards.clear()
            self.already_done = True
        
        return state, reward, done, info

    def reset(self, **kwargs):
        self.state_t = self.env.reset(**kwargs)
        self.reward_t = 0.
        self.already_done = False
        return self.state_t