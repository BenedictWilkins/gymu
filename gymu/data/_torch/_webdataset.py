#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" 
   Created on 17-02-2022
"""
__author__ = "Benedict Wilkins"
__email__ = "benrjw@gmail.com"
__status__ = "Development"

import logging
Logger = logging.getLogger('gymu')

import webdataset as wb
import numpy as np
import torch
import math

from tqdm.auto import tqdm
from torch.utils.data import IterableDataset, TensorDataset, DataLoader

from collections.abc import Iterable
from typing import List, Union, Dict, Any, Callable


from ...typing import Mode
from ...utils import overload
from ...iter import Iterator, episode

from ... import mode as m


from . import iterators


__all__ = ("dataset",)


"""
class GymuShorthands:

    @property
    def gymu(self):
        return _GymuShorthands(self)

# patch the wb.Process class to inherit from our custom Shorthands
wb.Processor.__bases__ += (GymuShorthands,) 
#type(wb.Processor.__name__ + "WithGymuShorthands", (wb.Processor, GymuShorthands), {})
"""

class GymuShorthands:

    # access like dataset.gymu.decode
    def decode(self, keep_meta=False):
        return self.then(iterators.decode, keep_meta=keep_meta)
    
    def mode(self, mode, ignore_last=True):
        return self.then(iterators.mode, mode, ignore_last=ignore_last)

    def keep(self, key, *keys): #[STATE, NEXT_STATE, ACTION, REWARD, DONE, INFO]
        return self.then(iterators.keep, keys=[key, *keys])

    def discard(self, key, *keys):
        return self.then(iterators.discard, keys=[key, *keys])
    
    def window(self, window_size=2, **kwargs):
        return self.then(iterators.window, window_size=window_size, **kwargs)

    def unpack_info(self):
        return self.then(iterators.unpack_info)
    
    def numpy(self):
        return self.then(iterators.numpy)

    def mask(self, **mask : Union[slice, np.ndarray]):
        return self.then(iterators.mask, mask)

    def to_tensor_dataset(self, num_workers : int = 0, show_progress : bool = False, order : List[str] = None): # WARNING YOU MIGHT RUN OUT OF MEMORY ;)
        import torch.multiprocessing
        torch.multiprocessing.set_sharing_strategy('file_system')                                  
        source = DataLoader(self, batch_size=512, shuffle=False, num_workers=num_workers, drop_last=False)   
        source = source if not show_progress else tqdm(source, desc="Loading Tensor Dataset")                                                                          
        source = iter(source)
        tensors = [[x] for x in next(source)]
        #print([(type(x[0])) for x in tensors])
        for z in tensors: # validate input... 
            if not torch.is_tensor(z[0]):
                raise ValueError(f"Expected torch.Tensor but found {type(z[0])}, choose a gymu.mode or convert to tuple before converting to a TensorDataset for example: 'dataset.decode().mode(gymu.sa).to_tensor_dataset()'.")
        for x in source:
            for z,t in zip(x, tensors):
                t.append(z)
        tensors = [torch.cat(z,dim=0) for z in tensors]  # TODO this uses double memory... perhaps we need to specify a max size?                                                     
        return TensorDataset(*tensors) 

class Composable(wb.Composable):

    def then(self, f, *args, **kw):
        #print("THEN")
        assert callable(f)
        assert "source" not in kw
        return Processor(self, f, *args, **kw)

class Processor(GymuShorthands, Composable, wb.Processor):
    pass

class _WebDatasetIterable(Processor): # wrapper for IterableDataset...
   
    def __init__(self, iterator):
        self.iterator = iterator
    
    def __iter__(self):
        return iter(self.iterator)

class _WebDatasetEnvironmentIterable(Processor):

    def __init__(self, env : Callable, policy : Callable, mode : Mode = m.sa, max_episode_length : int = 4096, num_episodes : int = 1):
        self.env = env
        self.policy = policy
        self._mode = mode
        self._max_episode_length = max(1, max_episode_length)
        self._num_episodes = max(1, num_episodes)

    def __iter__(self):
        # the process is the same for each worker if using multiprocess
        env = self.env()
        policy = self.policy() # TODO if this should be shared accross workers things are more complicated
        iterator = Iterator(env, policy, max_length=self._max_episode_length, mode=self._mode)
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:  # single-process data loading, return the full iterator
            for _ in range(self._num_episodes):
                yield from iterator
        else: 
            n = math.ceil(self._num_episodes / worker_info.num_workers)
            dif = (n * worker_info.num_workers ) - self._num_episodes 
            if worker_info.id < dif:
                n -= 1
            for _ in range(n):
                yield from iterator
                
@overload
def dataset(*args, **kwargs):
    pass 

@dataset.args(List)
def dataset(path: List, **kwargs):
    path = [str(p) for p in path]
    return _WebDatasetIterable(wb.WebDataset(path, **kwargs))

@dataset.args(str)
def dataset(path : str, **kwargs):
    return _WebDatasetIterable(wb.WebDataset(path, **kwargs))

@dataset.args(Iterable)
def dataset(iterator : Iterable, **kwargs):
    return _WebDatasetIterable(iterator, **kwargs).map(lambda x: dict(**x))

@dataset.args(Callable, Callable)
def dataset(env : Callable, policy : Callable, mode : Mode = m.sa, max_episode_length : int = 4096, num_episodes : int = 1):
    return _WebDatasetEnvironmentIterable(env, policy, mode=mode, max_episode_length=max_episode_length, num_episodes=num_episodes )