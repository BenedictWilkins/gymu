#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
 Created on 18-02-2021 09:50:57

 [Description]
"""
__author__ ="Benedict Wilkins"
__email__ = "benrjw@gmail.com"
__status__ ="Development"

from typing import Tuple, List, Callable, Union, Optional
from abc import abstractmethod, ABC

import numpy as np
import torch

from .. import mode as Mode

def identity(x):
    return x

class EpisodeBase(ABC):
    """ Base class for an episode. """
    @abstractmethod
    def get(self):
        pass 

class Episode(EpisodeBase):
    """ 
        Wrapper class for an episode.
    """ 

    def __init__(self, episode):
        self._episode = episode

    def get(self):
        return self._episode

class SplitEpisode(EpisodeBase):
    """ 
        In the case that an episode may spans multiple files, this enables them to 
        be properly loaded with numpy's mmap by concatinating on the fly. `chunks` is 
        expected to be an ordered list of episode chunks.

        For example, an episode consisting of only states: (gymu.mode.s)

        chunks[0] = mode(s=[s_0,s_1,s_2])
        chunks[1] = mode(s=[s_3,s_4,s_5,s_6])]
        ...

        if numpy's mmap is being used, the episode will be loaded into memory on use 
        of `get`.
    """
    def __init__(self, *chunks):
        super().__init__()
        self._chunks = chunks

    def get(self):
        return tuple(np.concatenate(x, axis=0) for x in zip(*self._chunks))

class EpisodeDataset(torch.utils.data.Dataset): # should be used with SplitEpisode and Episode below

    def __init__(self,  episodes : List[EpisodeBase],
                        transforms : List[Callable] = []):
        super().__init__()
    
        self._episodes = episodes
        self._transforms = transforms

    def _transform(self, episode): # transform each part of the episode using `transforms`
        for transform in self._transforms:
            episode = transform(episode)
        return episode

    def __getitem__(self, i):
        episode = self._episodes[i].get()
        return self._transform(episode)
    
    def __len__(self):
        return len(self._episodes)

""" # TODO remove? refactor? 
class GymIterableDataset(IterableDataset):

    def __init__(self, env, policy=None, mode=mode.s, workers=1, sync=True):
        super().__init__()
        if isinstance(env, str):
            env = lambda : gym.make(env)
        if not callable(env):
            raise ValueError("Supplied `env` {0} must be a callable with no arguments.".format(env))

        if sync: 
            self.env = SyncVectorEnv([env] * workers)
        else:
            self.env = AsyncVectorEnv([env] * workers)
        self.sync = sync
        self.workers = workers

        self.iterator = iterator(self.env, policy=policy, mode=mode)

    def __iter__(self):
        for n_entry in self.iterator:
            for entry in zip(*n_entry.tuple()):
                yield entry
"""