#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
    Created on 16-09-2020 13:13:31

    [Description]
"""
__author__ = "Benedict Wilkins"
__email__ = "benrjw@gmail.com"
__status__ = "Development"

import gym

import logging
Logger = logging.getLogger("gymu")
Logger.setLevel(logging.INFO) 

from . import iter
from . import mode
from . import policy
from . import spaces
from . import wrappers
from . import utils
from . import data
from . import typing

from .iter import episode, episodes, iterator, intercept

__all__ = ('iter', 'iterator', 'mode', 'policy', 'spaces', 'wrappers', 'utils', 'data', 'intercept', 'typing')


def make(env_id, **kwargs):
    return gym.make(env_id, **kwargs)




'''
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
'''



