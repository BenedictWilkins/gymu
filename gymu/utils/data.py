#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
 Created on 18-02-2021 09:50:57

 [Description]
"""
__author__ ="Benedict Wilkins"
__email__ = "benrjw@gmail.com"
__status__ ="Development"

import itertools
from gymu.iterators.base import episode, episodes, vectorize
from typing import Tuple, List, Callable, Union, Optional
from abc import abstractmethod, ABC

import gym
import numpy as np
import torch

from .. import mode
from ..iterators import stream, StreamWorker, vectorize, EpisodesWorker 

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

class LazyEpisode(EpisodeBase):

    def __init__(self, episode):
        self._episode = episode

    def get(self):
        return [x[...] for x in self._episode]

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

class _MultiBuffer: # torch Dataset backend

    def __init__(self, meta, device=None):
        self.device = device
        print(meta)
        self._buffer = [torch.empty(s, dtype=d, device=device) for s,d in meta]
    
    def __getitem__(self, i):
        return [x[i] for x in self._buffer]

    def __setitem__(self, i, value):
        # assume v is a numpy array ? 
        for x, v in zip(self._buffer, value):
            if isinstance(v, np.ndarray): 
                x[i] = torch.from_numpy(v).to(self.device, non_blocking=True)
            else: # is a float, int etc ? 
                x[i] = v

    def __len__(self):
        return len(self._buffer[0])



class IterableDataset(torch.utils.data.IterableDataset):

    def __init__(self, env, policy=None, mode=mode.sar, n=-1, max_episode_length=1000, num_environments=1):
        super().__init__()
        self.n = n

        if num_environments > 1:
            # check if ray is initialised...
            import ray # not good practice.. but its fine, this object will not be created very often !
            try: ray.init()
            except: pass # already initialised ?
            workers = [StreamWorker.remote(env, policy=policy, mode=mode, max_episode_length=max_episode_length) for _ in range(num_environments)]
            self._iter = ray.util.iter.from_actors(workers).gather_async() 
        else:
            self._iter = stream(env, policy=policy, mode=mode, max_episode_length=max_episode_length)    
        
    def __iter__(self):
        if self.n < 0: 
            for x in self._iter:
                yield x.tuple()
        else:
            for x in itertools.islice(self._iter, 0, self.n):
                yield x.tuple()

class IterableEpisodeDataset(torch.utils.data.IterableDataset):

    def __init__(self, env, policy=None, mode=mode.sar, n=-1, max_episode_length=1000, num_environments=1):
        self.n = n

        if num_environments > 1:
            # check if ray is initialised...
            import ray # not good practice.. but its fine, this object will not be created very often !
            try: ray.init()
            except: pass # already initialised ?
            self.workers = [EpisodesWorker.remote(env, policy=policy, mode=mode, n=-1, max_length=max_episode_length) for _ in range(num_environments)]
            self._iter = ray.util.iter.from_actors(self.workers).gather_async() 
        else:
            self.workers = [] # no workers...
            self._iter = episodes(env, policy=policy, mode=mode, n=-1, max_length=max_episode_length)    
        self.closed = False

    def __iter__(self):
        if self.closed:
            raise StopIteration

        if self.n < 0: 
            for x in self._iter:
                yield x.tuple()
        else:
            for x in itertools.islice(self._iter, 0, self.n):
                yield x.tuple()

    def close(self):
        if len(self.workers) > 0:
            import ray
            for worker in self.workers:
                ray.kill(worker)

class ReplayDataset(torch.utils.data.Dataset):

    def __init__(self, env, policy=None, mode=mode.sar, n=-1, max_episode_length=1000, num_environments=1, au=128, device="cpu"):
        self._iter = iter(IterableDataset(env, policy=policy, mode=mode, n=-1, max_episode_length=max_episode_length, num_environments=num_environments))
        
        # create data buffer
        x = next(self._iter) # get one sample to build the buffer 
        def get_dtype(i):
            if isinstance(i, np.ndarray):
                return torch.from_numpy(i).dtype
            else:
                return type(i)
        buffer_meta = [((n, *np.array(y).shape), get_dtype(y)) for y in x]
        self._buffer = _MultiBuffer(buffer_meta, device=device)
        self._buffer[0] = x
        # fill the buffer
        for i in range(1, len(self._buffer)):
            self._buffer[i] = next(self._iter)

        self._au = au # how many __getitem__ calls before grabbing new data from _iter
        self.__aui = 0 # how many calls to __getitem__
        self.__ui = 0 # update index (which element in the buffer to replace with new data)

    def __getitem__(self, i):
        x = self._buffer[i] 
        self.__aui += 1 
        if self.__aui >= self._au: 
            self._buffer[self.__ui] = next(self._iter)
            self.__ui = 0
        return x

    def __len__(self):
        return len(self._buffer)

    
